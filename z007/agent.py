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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, get_type_hints

import anyio
import boto3
from boto3.session import Session

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails"""


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, model_id: str, params: dict[str, Any]):
        self.model_id = model_id
        self.params = params

    @abstractmethod
    async def run_conversation(
        self,
        prompt: str,
        conversation_history: list[dict[str, Any]] | None,
        system_prompt: str | None,
        tool_registry: "ToolRegistry",
        max_turns: int,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Run a conversation with the LLM provider"""
        ...


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider"""

    def __init__(self, model_id: str, params: dict[str, Any]):
        super().__init__(model_id, params)
        if not can_access_bedrock():
            raise RuntimeError("Cannot access AWS Bedrock -- please set up AWS credentials")
        self.client = Session().client("bedrock-runtime")

    async def run_conversation(
        self,
        prompt: str,
        conversation_history: list[dict[str, Any]] | None,
        system_prompt: str | None,
        tool_registry: "ToolRegistry",
        max_turns: int,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Run conversation with Bedrock"""
        # Start with provided conversation history or empty list
        messages = conversation_history.copy() if conversation_history else []

        # Add user message to conversation
        messages.append({"role": "user", "content": [{"text": prompt}]})
        responses = []

        # Get tools in the format Bedrock expects
        available_tools_raw = tool_registry.get_tool_specs()
        available_tools = [{"toolSpec": tool["toolSpec"]} for tool in available_tools_raw]

        for turn_num in range(max_turns):
            tool_config = {"tools": available_tools, "toolChoice": {"any": {}}}

            try:
                # Prepare the converse parameters
                converse_params = {
                    "modelId": self.model_id,
                    "messages": messages,  # type: ignore
                    "toolConfig": tool_config,  # type: ignore
                }

                # Add system prompt as a separate parameter if provided
                if system_prompt:
                    converse_params["system"] = [{"text": system_prompt}]

                # Make bedrock call
                response = await anyio.to_thread.run_sync(  # type: ignore
                    lambda: self.client.converse(**converse_params)
                )
                # Normalize to OpenAI format for consistency
                normalized_response = self._normalize_bedrock_response_to_openai(response)
                responses.append(normalized_response)

                # Check if we need to handle tool calls (use raw response for processing)
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
                            results = await _execute_tools_concurrently(tool_registry, tool_calls)

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

    def _normalize_bedrock_response_to_openai(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert Bedrock response format to OpenAI-style format for consistency"""
        try:
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            # Extract text content and tool calls
            text_content = ""
            tool_calls = []

            for item in content:
                if "text" in item:
                    text_content = item["text"]
                elif "toolUse" in item:
                    tool_use = item["toolUse"]
                    tool_calls.append(
                        {
                            "id": tool_use["toolUseId"],
                            "type": "function",
                            "function": {"name": tool_use["name"], "arguments": json.dumps(tool_use["input"])},
                        }
                    )

            # Build OpenAI-style response
            openai_message = {"role": "assistant", "content": text_content}

            if tool_calls:
                openai_message["tool_calls"] = tool_calls

            return {"choices": [{"message": openai_message, "finish_reason": "tool_calls" if tool_calls else "stop"}]}

        except Exception as e:
            logger.error(f"Error normalizing Bedrock response: {e}")
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "Error processing response"}, "finish_reason": "stop"}
                ]
            }

    async def cleanup(self):
        """Cleanup Bedrock resources (no-op for Bedrock)"""


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider using httpx"""

    def __init__(self, model_id: str, params: dict[str, Any]):
        super().__init__(model_id, params)
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx package is required for OpenAI provider. Install with: uv add httpx")

        self.base_url = params.get("base_url", "http://127.0.0.1:1234/v1")
        self.api_key = params.get("api_key", "dummy-key")
        self.client = httpx.AsyncClient(
            base_url=self.base_url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30.0
        )

    async def run_conversation(
        self,
        prompt: str,
        conversation_history: list[dict[str, Any]] | None,
        system_prompt: str | None,
        tool_registry: "ToolRegistry",
        max_turns: int,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Run conversation with OpenAI-compatible API using httpx"""
        # Convert conversation history to OpenAI format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Convert Bedrock format to OpenAI format
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, list) and len(content) > 0:
                        if "text" in content[0]:
                            messages.append({"role": "user", "content": content[0]["text"]})
                        else:
                            # Handle tool results
                            messages.append({"role": "user", "content": str(content)})
                elif msg["role"] == "assistant":
                    content = msg["content"]
                    if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                        messages.append({"role": "assistant", "content": content[0]["text"]})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        responses = []

        # Get tools in OpenAI format
        tools = self._convert_tools_to_openai_format(tool_registry.get_tool_specs())

        for turn_num in range(max_turns):
            try:
                # Prepare request payload
                payload = {
                    "model": self.model_id,
                    "messages": messages,
                }
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = "auto"

                # Make httpx request to chat completions endpoint
                response = await self.client.post("/chat/completions", json=payload, timeout=300)
                response.raise_for_status()

                response_data = response.json()
                responses.append(response_data)

                # Extract message from response
                if "choices" not in response_data or not response_data["choices"]:
                    break

                message = response_data["choices"][0]["message"]
                content = message.get("content", "")
                messages.append({"role": "assistant", "content": content})

                # Check for tool calls
                tool_calls_data = message.get("tool_calls")
                if tool_calls_data:
                    tool_calls = []
                    for tool_call in tool_calls_data:
                        function_data = tool_call["function"]
                        tool_calls.append(
                            (function_data["name"], json.loads(function_data["arguments"]), tool_call["id"])
                        )

                    if tool_calls:
                        # Execute tools
                        results = await _execute_tools_concurrently(tool_registry, tool_calls)

                        # Convert results to OpenAI format
                        for i, (_, _, tool_call_id) in enumerate(tool_calls):
                            result = results[i]
                            content = result["toolResult"]["content"][0]["text"]
                            messages.append({"role": "tool", "content": content, "tool_call_id": tool_call_id})
                        # Continue to get LLM response to tool results
                        continue

                    break

                # No tool calls, conversation is complete
                break

            except Exception as e:
                logger.error(f"Error in conversation turn {turn_num}: {e}")
                break

        # Convert back to Bedrock format for consistency
        bedrock_messages = []
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                bedrock_messages.append({"role": msg["role"], "content": [{"text": msg.get("content", "")}]})

        return responses, bedrock_messages

    async def cleanup(self):
        """Cleanup httpx client"""
        if hasattr(self, "client"):
            await self.client.aclose()

    def _convert_tools_to_openai_format(self, bedrock_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Bedrock tool format to OpenAI format"""
        openai_tools = []
        for tool in bedrock_tools:
            tool_spec = tool["toolSpec"]
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
            )
        return openai_tools


async def _execute_tools_concurrently(
    tool_registry: "ToolRegistry", tool_calls: list[tuple[str, dict[str, Any], str]]
) -> list[dict[str, Any]]:
    """Execute all tools concurrently and return results"""

    async def execute_single_tool(name: str, input_data: dict[str, Any], use_id: str) -> dict[str, Any]:
        try:
            result = await tool_registry.execute(name, input_data)
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

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Get all tools as tool specifications"""
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
    """Agent with tool and MCP support"""

    def __init__(
        self,
        model_id: str,
        provider: str = "bedrock",
        provider_params: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        mcp_config: dict[str, Any] | None = None,
        max_turns: int = 5,
    ):
        # Create provider
        params = provider_params or {}
        if provider == "bedrock":
            self._provider = BedrockProvider(model_id, params)
        elif provider == "openai":
            self._provider = OpenAIProvider(model_id, params)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.system_prompt = system_prompt
        self.max_turns = max_turns

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
        """Extract final answer from response (OpenAI format)"""
        try:
            # Handle OpenAI format
            choices = response.get("choices")
            if choices:
                choice = choices[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()

            # Fallback: try Bedrock format for backward compatibility
            content = response.get("output", {}).get("message", {}).get("content", [])
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    if isinstance(text, str):
                        return text
            return "No final answer found"
        except Exception:
            return "No final answer found"

    async def run_conversation(
        self, prompt: str, conversation_history: list[dict[str, Any]] | None = None
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Run conversation with tool support - delegates to provider

        Args:
            prompt: The user's input message
            conversation_history: Optional prior conversation history

        Returns:
            Tuple of (responses from LLM, updated conversation history)
        """
        return await self._provider.run_conversation(
            prompt, conversation_history, self.system_prompt, self._tool_registry, self.max_turns
        )

    async def cleanup(self):
        """Cleanup provider resources"""
        if hasattr(self._provider, "cleanup"):
            await self._provider.cleanup()


def get_called_tools(responses: list[Any]) -> list[str]:
    """Get list of called tools (OpenAI format only)"""
    tools: list[str] = []
    for response in responses:
        try:
            choices = response.get("choices", [])
            for choice in choices:
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])
                for tool_call in tool_calls:
                    function_data = tool_call.get("function", {})
                    tool_name = function_data.get("name")
                    if tool_name:
                        tools.append(str(tool_name))
        except Exception:
            continue
    return tools


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


def can_access_bedrock() -> bool:
    """Checks if Boto3 credentials can call Bedrock, returning a boolean."""
    try:
        _ = boto3.Session().client("bedrock-runtime")
        return True
    except Exception as e:
        logger.info(f"Unexpected Bedrock access error: {e}")
        return False
