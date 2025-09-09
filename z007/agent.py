#!/usr/bin/env python3
"""
Agent module containing ToolRegistry and Agent classes for LLM integration.
Supports multiple providers including AWS Bedrock, with tool and MCP support.
"""

import inspect
import json
import logging
import select
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, get_type_hints

import anyio
from boto3.session import Session

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails"""


class ToolRegistry:
    """Streamlined registry for both local and MCP tools with context manager support"""

    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., Any]] = {}  # {tool_name: function}
        self.tool_metadata: dict[str, dict[str, Any]] = {}  # {tool_name: metadata}
        self.mcp_servers: dict[str, subprocess.Popen[str]] = {}  # {server_name: process}
        self.mcp_tools: dict[str, str] = {}  # {tool_name: server_name}

    def register(self, func: Callable[..., Any], **metadata: Any):
        """Register a function as a tool"""
        self.tools[func.__name__] = func
        if metadata:
            self.tool_metadata[func.__name__] = metadata

    def load_mcp_config(self, config_path: str):
        """Load MCP servers from config file"""
        try:
            if not Path(config_path).exists():
                return self

            with open(config_path) as f:
                config = json.load(f)

            self.load_mcp_config_dict(config)
        except Exception as e:
            logger.error(f"MCP config error: {e}")
            raise ToolExecutionError(f"Failed to load MCP config: {e}")

    def load_mcp_config_dict(self, config: dict[str, Any]):
        """Load MCP servers from config dictionary"""
        try:
            for name, cfg in config.get("servers", {}).items():
                command = cfg.get("command", [])
                args = cfg.get("args", [])
                env_vars = cfg.get("env", {})

                if command:
                    full_cmd = [command, *args] if isinstance(command, str) else command + args
                    self._start_mcp_server(name, full_cmd, env_vars)
        except Exception as e:
            logger.error(f"MCP config error: {e}")
            raise ToolExecutionError(f"Failed to load MCP config: {e}")

    def _start_mcp_server(self, name: str, command: list[str], env_vars: dict[str, str] | None = None) -> None:
        """Start MCP server and load tools"""
        import os

        try:
            # Expand shell variables in command arguments using os.path.expandvars
            expanded_command = []
            for arg in command:
                expanded_arg = os.path.expandvars(arg)
                expanded_command.append(expanded_arg)

            # Prepare environment with config variables
            env = os.environ.copy()
            if env_vars:
                for key, value in env_vars.items():
                    if value.startswith("${env:") and value.endswith("}"):
                        env_var = value[6:-1]
                        if env_var in os.environ:
                            env[key] = os.environ[env_var]
                    else:
                        env[key] = value

            logger.info(f"Starting MCP server '{name}' with command: {' '.join(expanded_command)}")
            process = subprocess.Popen(
                expanded_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            time.sleep(0.5)

            if (return_code := process.poll()) is not None:
                error_msg = f"MCP '{name}' failed to start with return code {return_code}"
                logger.error(error_msg)
                raise ToolExecutionError(error_msg)

            # Check if stdin is available
            if process.stdin is None or process.stdout is None:
                error_msg = f"MCP '{name}' stdin or stdout not available"
                logger.error(error_msg)
                raise ToolExecutionError(error_msg)

            self.mcp_servers[name] = process

            # Send MCP protocol messages
            msgs = [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "clientInfo": {"name": "z007-agent", "version": "1.0.0"},
                    },
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            ]

            for msg in msgs:
                process.stdin.write(json.dumps(msg) + "\n")
            process.stdin.flush()

            # Read responses - handle initialization response and tools list response
            tools_count = 0
            start_time = time.time()
            initialization_received = False

            while time.time() - start_time < 10:  # 10 second timeout
                if select.select([process.stdout], [], [], 0.1)[0]:
                    try:
                        response = json.loads(process.stdout.readline())

                        # Handle initialization response
                        if not initialization_received and response.get("id") == 1 and "result" in response:
                            initialization_received = True
                            logger.debug(f"MCP '{name}' initialized successfully")
                            continue

                        # Handle tools list response
                        if response.get("id") == 2 and "result" in response and "tools" in response.get("result", {}):
                            for tool in response["result"]["tools"]:
                                tool_name = tool["name"]
                                self.mcp_tools[tool_name] = name
                                self.tool_metadata[tool_name] = {
                                    "description": tool.get("description", f"MCP: {tool_name}"),
                                    "mcp_schema": tool.get("inputSchema", {}),
                                    "is_mcp": True,
                                }
                                tools_count += 1
                            logger.info(f"Loaded {tools_count} tools from MCP '{name}'")
                            return

                        # Handle errors
                        if "error" in response:
                            error_msg = f"MCP '{name}' error: {response['error']}"
                            logger.error(error_msg)
                            raise ToolExecutionError(error_msg)

                        # Log other responses for debugging
                        logger.debug(f"MCP '{name}' response: {response}")

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"MCP '{name}' JSON decode error: {e}")
                        continue

                if process.poll() is not None:
                    error_msg = f"MCP '{name}' process terminated unexpectedly"
                    logger.error(error_msg)
                    raise ToolExecutionError(error_msg)

            error_msg = f"MCP '{name}' timeout waiting for tools list"
            logger.error(error_msg)
            raise ToolExecutionError(error_msg)
        except Exception as e:
            logger.error(f"MCP '{name}' error: {e}")
            if not isinstance(e, ToolExecutionError):
                raise ToolExecutionError(f"Failed to start MCP server '{name}': {e}")
            raise

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute tool (local or MCP) asynchronously"""
        try:
            if tool_name in self.mcp_tools:
                return await self._execute_mcp_async(tool_name, tool_input)
            if tool_name in self.tools:
                func = self.tools[tool_name]
                sig = inspect.signature(func)
                kwargs = {p: tool_input.get(p) for p in sig.parameters if p in tool_input}

                # Run sync function in thread using AnyIO
                def call_with_kwargs() -> Any:
                    return func(**kwargs)

                result = await anyio.to_thread.run_sync(call_with_kwargs)  # type: ignore
                return str(result) if result is not None else ""
            raise ToolExecutionError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            if isinstance(e, ToolExecutionError):
                raise
            raise ToolExecutionError(f"Tool {tool_name} failed: {e}")

    def _execute_mcp_sync(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Synchronous MCP execution for thread pool"""
        try:
            server_name = self.mcp_tools[tool_name]
            process = self.mcp_servers[server_name]

            # Check if stdin and stdout are available
            if process.stdin is None or process.stdout is None:
                raise ToolExecutionError(f"Process streams not available for {tool_name}")

            request = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),  # Use timestamp for unique ID
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": tool_input},
            }

            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()

            # Simple timeout response reading
            start_time = time.time()
            while time.time() - start_time < 10:
                if select.select([process.stdout], [], [], 0.1)[0]:
                    try:
                        response = json.loads(process.stdout.readline())
                        if "result" in response:
                            content = response["result"].get("content", [])
                            if content:
                                return str(content[0].get("text", content[0]))
                            return "No content"
                        if "error" in response:
                            raise ToolExecutionError(f"MCP Error: {response['error'].get('message', 'Unknown')}")
                    except (json.JSONDecodeError, KeyError):
                        continue
                if process.poll() is not None:
                    raise ToolExecutionError(f"MCP server for {tool_name} terminated")
            raise ToolExecutionError(f"Timeout executing {tool_name}")
        except Exception as e:
            if isinstance(e, ToolExecutionError):
                raise
            raise ToolExecutionError(f"MCP execution failed for {tool_name}: {e}")

    async def _execute_mcp_async(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute MCP tool asynchronously"""
        return await anyio.to_thread.run_sync(self._execute_mcp_sync, tool_name, tool_input)  # type: ignore

    def get_bedrock_specs(self) -> list[dict[str, Any]]:
        """Get all tools as Bedrock specifications"""
        specs: list[dict[str, Any]] = []

        # Local tools
        for name, func in self.tools.items():
            metadata = self.tool_metadata.get(name, {})
            description = metadata.get("description") or func.__doc__ or f"Tool: {name}"

            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            properties: dict[str, dict[str, str]] = {}
            required: list[str] = []

            for param_name, param in sig.parameters.items():
                param_type = type_hints.get(param_name, str)
                json_type = "string"  # Simple fallback
                if param_type is int:
                    json_type = "integer"
                elif param_type is float:
                    json_type = "number"
                elif param_type is bool:
                    json_type = "boolean"

                properties[param_name] = {
                    "type": json_type,
                    "description": f"Parameter: {param_name}",
                }
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            specs.append(
                {
                    "toolSpec": {
                        "name": name,
                        "description": description,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                            }
                        },
                    }
                }
            )

        # MCP tools
        for name in self.mcp_tools:
            metadata = self.tool_metadata.get(name, {})
            mcp_schema = metadata.get("mcp_schema", {})

            specs.append(
                {
                    "toolSpec": {
                        "name": name,
                        "description": metadata.get("description", f"MCP: {name}"),
                        "inputSchema": {"json": mcp_schema or {"type": "object", "properties": {}}},
                    }
                }
            )

        return specs

    def cleanup(self) -> None:
        """Cleanup MCP servers"""
        for server_name, process in self.mcp_servers.items():
            try:
                process.terminate()
                process.wait(timeout=2)
                logger.info(f"Terminated MCP server: {server_name}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing MCP server: {server_name}")
                process.kill()
            except Exception as e:
                logger.error(f"Error terminating MCP server {server_name}: {e}")
        self.mcp_servers.clear()
        self.mcp_tools.clear()


class Agent:
    """Fast Micro Agent with tool support - supports multiple LLM providers"""

    def __init__(
        self,
        model_id: str,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        mcp_config: dict[str, Any] | None = None,
        max_turns: int = 5,
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.bedrock_runtime_client = Session().client("bedrock-runtime")

        # Create internal tool registry
        self._tool_registry = ToolRegistry()

        # Register provided tools
        if tools:
            for tool in tools:
                self._tool_registry.register(tool)

        # Load MCP configuration if provided
        if mcp_config:
            self._tool_registry.load_mcp_config_dict(mcp_config)

    async def __aenter__(self) -> "Agent":
        """Async context manager entry"""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - cleanup resources"""
        try:
            self._tool_registry.cleanup()
        except Exception as e:
            logger.error(f"Error during Agent cleanup: {e}")

    def __enter__(self) -> "Agent":
        """Sync context manager entry (for backward compatibility)"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Sync context manager exit - cleanup resources"""
        try:
            self._tool_registry.cleanup()
        except Exception as e:
            logger.error(f"Error during Agent cleanup: {e}")

    def get_tool_counts(self) -> tuple[int, int, int]:
        """Get tool counts (local, mcp_servers, mcp_tools)"""
        return (
            len(self._tool_registry.tools),
            len(self._tool_registry.mcp_servers),
            len(self._tool_registry.mcp_tools),
        )

    def get_tool_names(self) -> tuple[list[str], list[str]]:
        """Get tool names (local_tools, mcp_tools)"""
        return (
            list(self._tool_registry.tools.keys()),
            list(self._tool_registry.mcp_tools.keys()),
        )

    async def run(self, prompt: str) -> str:
        """Run a single conversation and return the final answer"""
        responses, _ = await self.run_conversation(prompt)
        return Agent.extract_final_answer(responses[-1]) if responses else "No response"

    @staticmethod
    def extract_final_answer(response: Any) -> str:
        """Extract final answer from response"""
        try:
            content = response.get("output", {}).get("message", {}).get("content", [])
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    if isinstance(text, str):
                        return text
            return "No final answer found"
        except Exception:
            return "No final answer found"

    async def _execute_tools_concurrently(
        self, tool_calls: list[tuple[str, dict[str, Any], str]]
    ) -> list[dict[str, Any]]:
        """Execute all tools concurrently and return results"""

        async def execute_single_tool(name: str, input_data: dict[str, Any], use_id: str) -> dict[str, Any]:
            try:
                result = await self._tool_registry.execute(name, input_data)
                return {"toolResult": {"toolUseId": use_id, "content": [{"text": result}]}}
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return {
                    "toolResult": {
                        "toolUseId": use_id,
                        "content": [{"text": f"Error: {e!s}"}],
                        "status": "error",
                    }
                }

        # Execute all tools concurrently
        results: list[dict[str, Any]] = [{}] * len(tool_calls)
        async with anyio.create_task_group() as tg:
            for i, (name, input_data, use_id) in enumerate(tool_calls):

                async def run_tool(
                    idx: int = i,
                    n: str = name,
                    d: dict[str, Any] = input_data,
                    u: str = use_id,
                ) -> None:
                    results[idx] = await execute_single_tool(n, d, u)

                tg.start_soon(run_tool)
        return results

    async def run_conversation(
        self, prompt: str, conversation_history: list[dict[str, Any]] | None = None
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Run conversation with tool support - async tool execution using AnyIO

        Args:
            prompt: The user's input message
            conversation_history: Optional prior conversation history

        Returns:
            Tuple of (responses from LLM, updated conversation history)
        """
        # Start with provided conversation history or empty list
        messages = conversation_history.copy() if conversation_history else []

        # Add user message to conversation
        messages.append({"role": "user", "content": [{"text": prompt}]})
        responses = []

        # Get tools in the format Bedrock expects
        available_tools_raw = self._tool_registry.get_bedrock_specs()
        available_tools = [{"toolSpec": tool["toolSpec"]} for tool in available_tools_raw]

        for turn_num in range(self.max_turns):
            tool_config = {"tools": available_tools, "toolChoice": {"any": {}}}

            try:
                # Prepare the converse parameters
                converse_params = {
                    "modelId": self.model_id,
                    "messages": messages,  # type: ignore
                    "toolConfig": tool_config,  # type: ignore
                }

                # Add system prompt as a separate parameter if provided
                if self.system_prompt:
                    converse_params["system"] = [{"text": self.system_prompt}]
                # Make bedrock call
                response = await anyio.to_thread.run_sync(  # type: ignore
                    lambda: self.bedrock_runtime_client.converse(**converse_params)
                )
                responses.append(response)

                # Check if we need to handle tool calls
                stop_reason = response["stopReason"]
                if stop_reason == "tool_use":
                    # Get the assistant's message with tool calls
                    output = response["output"]
                    assistant_msg = output["message"]

                    if assistant_msg:
                        # Add assistant message to conversation
                        messages.append(
                            {
                                "role": assistant_msg["role"],
                                "content": assistant_msg["content"],
                            }
                        )

                        # Extract tool calls
                        tool_calls = []
                        for item in assistant_msg["content"]:
                            if "reasoningContent" in item:
                                reasoning_content = item["reasoningContent"]
                                reasoning_text = reasoning_content["reasoningText"]
                                text = reasoning_text["text"]
                                print(f"Reasoning: {text}")
                            elif "toolUse" in item:
                                tool_use = item["toolUse"]
                                name = tool_use["name"]
                                input_data = tool_use["input"]
                                use_id = tool_use["toolUseId"]

                                if name and use_id:
                                    tool_calls.append((name, input_data, use_id))
                            else:
                                logger.warning(f"Unknown item in assistant message content: {item}")

                        if tool_calls:
                            # Execute all tools concurrently and collect results
                            results = await self._execute_tools_concurrently(tool_calls)

                            # Add tool results to conversation
                            messages.append({"role": "user", "content": results})
                        else:
                            logger.warning("No tool calls found despite tool_use stop reason")
                            break
                    else:
                        logger.debug(f"No assistant message found in response: {response}")
                        break
                else:
                    logger.debug(f"Conversation ended with stop reason: {stop_reason}")
                    break

            except Exception as e:
                logger.error(f"Error in conversation turn {turn_num}: {e}")
                break

        return responses, messages


def create_calculator_tool() -> Callable[..., str]:
    """Create a basic calculator tool function"""

    def calculator_tool(expression: str) -> str:
        """Calculator tool - performs mathematical calculations"""
        try:
            # Basic safety check
            if any(char in expression for char in ["import", "exec", "eval", "__"]):
                return "Error: Invalid expression"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return calculator_tool
