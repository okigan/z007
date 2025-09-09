#!/usr/bin/env python3
"""
Streamlined test to check if models call non-existent tools when overwhelmed.
Provides many tools and checks if the model tries to call tools that don't exist.
"""

import boto3
import json
import inspect
import logging
import subprocess
import select
import time
import colorlog
import anyio

from pathlib import Path
from typing import get_type_hints, Any, Callable

# Set up colored logging
colorlog.basicConfig(level=logging.INFO, format='%(log_color)s%(asctime)s - %(levelname)s - %(message)s')
logger = colorlog.getLogger(__name__)

MODEL_ID = "openai.gpt-oss-20b-1:0"

class ToolRegistry:
    """Streamlined registry for both local and MCP tools"""
    
    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., Any]] = {}  # {tool_name: function}
        self.tool_metadata: dict[str, dict[str, Any]] = {}  # {tool_name: metadata}
        self.mcp_servers: dict[str, subprocess.Popen[str]] = {}  # {server_name: process}
        self.mcp_tools: dict[str, str] = {}  # {tool_name: server_name}
    
    def register(self, func: Callable[..., Any], **metadata: Any) -> "ToolRegistry":
        """Register a function as a tool"""
        self.tools[func.__name__] = func
        if metadata:
            self.tool_metadata[func.__name__] = metadata
        return self
    
    def load_mcp_config(self, config_path: str) -> "ToolRegistry":
        """Load MCP servers from config file"""
        try:
            if not Path(config_path).exists():
                return self
            
            with open(config_path) as f:
                config = json.load(f)
            
            for name, cfg in config.get("servers", {}).items():
                command = cfg.get("command", [])
                args = cfg.get("args", [])
                if command:
                    full_cmd = [command] + args if isinstance(command, str) else command + args
                    self._start_mcp_server(name, full_cmd)
            
            return self
        except Exception as e:
            logger.error(f"MCP config error: {e}")
            return self
    
    def _start_mcp_server(self, name: str, command: list[str]) -> None:
        """Start MCP server and load tools"""
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            time.sleep(0.5)
            
            if (return_code := process.poll()) is not None:
                logger.error(f"MCP '{name}' failed to start with return code {return_code}")
                return

            # Check if stdin is available
            if process.stdin is None or process.stdout is None:
                logger.error(f"MCP '{name}' stdin or stdout not available")
                return
            
            self.mcp_servers[name] = process
            
            # Send MCP protocol messages
            msgs = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            ]
            
            for msg in msgs:
                process.stdin.write(json.dumps(msg) + "\n")
            process.stdin.flush()
            
            # Read tools response
            tools_count = 0
            start_time = time.time()
            while time.time() - start_time < 5:  # 5 second timeout
                if select.select([process.stdout], [], [], 0.1)[0]:
                    try:
                        response = json.loads(process.stdout.readline())
                        if "result" in response and "tools" in response.get("result", {}):
                            for tool in response["result"]["tools"]:
                                tool_name = tool["name"]
                                self.mcp_tools[tool_name] = name
                                self.tool_metadata[tool_name] = {
                                    "description": tool.get("description", f"MCP: {tool_name}"),
                                    "mcp_schema": tool.get("inputSchema", {}),
                                    "is_mcp": True
                                }
                                tools_count += 1
                            logger.info(f"Loaded {tools_count} tools from MCP '{name}'")
                            return
                    except (json.JSONDecodeError, KeyError):
                        continue
                if process.poll() is not None:
                    break
            
            logger.error(f"MCP '{name}' timeout")
        except Exception as e:
            logger.error(f"MCP '{name}' error: {e}")
    
    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute tool (local or MCP) asynchronously"""
        if tool_name in self.mcp_tools:
            return await self._execute_mcp_async(tool_name, tool_input)
        elif tool_name in self.tools:
            func = self.tools[tool_name]
            sig = inspect.signature(func)
            kwargs = {p: tool_input.get(p) for p in sig.parameters.keys() if p in tool_input}
            # Run sync function in thread using AnyIO  
            def call_with_kwargs() -> Any:
                return func(**kwargs)
            return await anyio.to_thread.run_sync(call_with_kwargs)  # type: ignore
        else:
            return f"Error: Unknown tool {tool_name}"
    
    def _execute_mcp(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute MCP tool"""
        try:
            server_name = self.mcp_tools[tool_name]
            process = self.mcp_servers[server_name]
            
            # Check if stdin and stdout are available
            if process.stdin is None or process.stdout is None:
                return f"Error: Process streams not available for {tool_name}"
            
            request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": tool_input}
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
                        elif "error" in response:
                            return f"MCP Error: {response['error'].get('message', 'Unknown')}"
                    except (json.JSONDecodeError, KeyError):
                        continue
                if process.poll() is not None:
                    break
            return f"Timeout: {tool_name}"
        except Exception as e:
            return f"Error: {e}"

    async def _execute_mcp_async(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute MCP tool asynchronously"""
        try:
            # Run the sync MCP communication using AnyIO
            return await anyio.to_thread.run_sync(self._execute_mcp_sync, tool_name, tool_input)  # type: ignore
        except Exception as e:
            return f"Error: {e}"

    def _execute_mcp_sync(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Synchronous MCP execution for thread pool"""
        try:
            server_name = self.mcp_tools[tool_name]
            process = self.mcp_servers[server_name]
            
            # Check if stdin and stdout are available
            if process.stdin is None or process.stdout is None:
                return f"Error: Process streams not available for {tool_name}"
            
            request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": tool_input}
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
                        elif "error" in response:
                            return f"MCP Error: {response['error'].get('message', 'Unknown')}"
                    except (json.JSONDecodeError, KeyError):
                        continue
                if process.poll() is not None:
                    break
            return f"Timeout: {tool_name}"
        except Exception as e:
            return f"Error: {e}"
    
    def get_bedrock_specs(self) -> list[dict[str, Any]]:
        """Get all tools as Bedrock specifications"""
        specs: list[dict[str, Any]] = []
        
        # Local tools
        for name, func in self.tools.items():
            metadata = self.tool_metadata.get(name, {})
            description = metadata.get('description') or func.__doc__ or f"Tool: {name}"
            
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
                
                properties[param_name] = {"type": json_type, "description": f"Parameter: {param_name}"}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            
            specs.append({
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": {"type": "object", "properties": properties, "required": required}}
                }
            })
        
        # MCP tools
        for name in self.mcp_tools.keys():
            metadata = self.tool_metadata.get(name, {})
            mcp_schema = metadata.get('mcp_schema', {})
            
            specs.append({
                "toolSpec": {
                    "name": name,
                    "description": metadata.get('description', f"MCP: {name}"),
                    "inputSchema": {"json": mcp_schema or {"type": "object", "properties": {}}}
                }
            })
        
        return specs
    
    def cleanup(self) -> None:
        """Cleanup MCP servers"""
        for name, process in self.mcp_servers.items():
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                pass
        self.mcp_servers.clear()
        self.mcp_tools.clear()


async def run_conversation(prompt: str, model_id: str, tool_registry_instance: ToolRegistry) -> list[dict[str, Any]]:
    """Run conversation with Bedrock - async tool execution using AnyIO"""
    bedrock = boto3.client("bedrock-runtime")
    
    try:
        messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": prompt}]}]
        responses: list[dict[str, Any]] = []
        
        # Get current tools (including any MCP tools loaded)
        available_tools = tool_registry_instance.get_bedrock_specs()
        
        for _ in range(5):  # Max 5 turns
            # Run bedrock call in thread using AnyIO
            def bedrock_call() -> Any:  # AWS SDK doesn't have precise return types
                return bedrock.converse(
                    modelId=model_id,
                    messages=messages,
                    toolConfig={"tools": available_tools, "toolChoice": {"any": {}}}
                )
            
            response = await anyio.to_thread.run_sync(bedrock_call)  # type: ignore
            responses.append(response)
            
            stop_reason = response.get('stopReason')
            if stop_reason == 'tool_use':
                # Handle tool calls
                assistant_msg = response.get('output', {}).get('message')
                if assistant_msg:
                    messages.append(assistant_msg)
                    content = assistant_msg.get('content', [])
                    
                    # Execute tools concurrently using AnyIO
                    tool_calls = []
                    for item in content:
                        if isinstance(item, dict) and 'toolUse' in item:
                            tool_use = item['toolUse']
                            tool_calls.append((
                                tool_use.get('name'),
                                tool_use.get('input', {}),
                                tool_use.get('toolUseId')
                            ))
                    
                    if tool_calls:
                        # Execute all tools concurrently using AnyIO task group
                        results = []  # Initialize results list
                        
                        async with anyio.create_task_group() as tg:
                            async def execute_and_collect(name: str, input_data: dict[str, Any], use_id: str) -> None:
                                result = await tool_registry_instance.execute(name, input_data)
                                results.append({
                                    "toolResult": {
                                        "toolUseId": use_id,
                                        "content": [{"text": result}]
                                    }
                                })
                            
                            # Start all tool executions concurrently
                            for name, input_data, use_id in tool_calls:
                                tg.start_soon(execute_and_collect, name, input_data, use_id)
                        
                        # All tasks are done, results contains all tool results
                        messages.append({"role": "user", "content": results})
                    else:
                        break
                else:
                    # log the issue
                    logger.debug(f"Assistant message not found in response: {response}")
                    break
            else:
                # log the unexpected stop reason, and whole response as one liner
                logger.debug(f"Unexpected stop reason: {stop_reason}, Full response: {response}")
                break
        
        return responses
    except Exception as e:
        return [{"error": str(e)}]

def extract_final_answer(response: dict[str, Any]) -> str:
    """Extract final answer from response"""
    try:
        content = response.get('output', {}).get('message', {}).get('content', [])
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                return item['text'][:200] + ("..." if len(item['text']) > 200 else "")
        return "No final answer found"
    except Exception:
        return "No final answer found"

def get_called_tools(responses: list[dict[str, Any]]) -> list[str]:
    """Get list of called tools"""
    tools: list[str] = []
    for response in responses:
        try:
            content = response.get('output', {}).get('message', {}).get('content', [])
            for item in content:
                if isinstance(item, dict) and 'toolUse' in item:
                    tool_name = item['toolUse'].get('name')
                    if tool_name and isinstance(tool_name, str):
                        tools.append(tool_name)
        except Exception:
            continue
    return tools

def create_tool_registry() -> ToolRegistry:
    """Create and configure the tool registry with all tools"""
    registry = ToolRegistry()
    
    # Simple tool functions
    def calculator_tool(expression: str) -> str:
        """Calculator tool - performs mathematical calculations"""
        try:
            # Basic safety check
            if any(char in expression for char in ['import', 'exec', 'eval', '__']):
                return "Error: Invalid expression"
            result: Any = eval(expression)  # eval can return Any type
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def noisy_text_generator(request: str = "test data") -> str:
        """Generate text with various non-alphanumeric characters"""
        import random
        import string
        
        # Generate random noise
        noise_chars = ''.join(random.choices(string.ascii_letters + string.digits + '!@#$%^&*()[]{}|\\:";\'<>?,./~`', k=50))
        unicode_chars = "¡™£¢∞§¶•ªº–≠œ∑´®†¥¨ˆøπ«åß∂ƒ©˙∆˚¬…æΩ≈ç√∫˜µ≤≥÷"
        
        return f"""NOISE_START_{noise_chars}_NOISE_MID
Generated text based on request: {request}
EXTRA_CHARS: {unicode_chars}
MORE_NOISE_{noise_chars}_END_NOISE
RANDOM_SYMBOLS: ◊Ω≈ç√∫˜µ≤≥÷æ…¬Ω≈ç√∫˜
{noise_chars}
FINAL_NOISE_BLOCK_{noise_chars}_COMPLETE"""

    # Register basic tools
    registry.register(calculator_tool)
    registry.register(noisy_text_generator)

    # Generate 50 dummy calculator tools
    NUM_TOOLS = 50
    for i in range(NUM_TOOLS):
        def make_tool(tool_id: int) -> Callable[[str], str]:
            def tool_func(expression: str) -> str:
                try:
                    # Basic safety check
                    if any(char in expression for char in ['import', 'exec', 'eval', '__']):
                        return "Error: Invalid expression"
                    result: Any = eval(expression)  # eval can return Any type
                    return str(result)
                except Exception as e:
                    return f"Error: {e}"
            tool_func.__name__ = f"tool_{tool_id}"
            tool_func.__doc__ = f"Calculator tool {tool_id} - performs mathematical calculations"
            return tool_func
        
        registry.register(make_tool(i))
    
    return registry


# Test cases
TEST_CASES = [
    "what is name on the https://github.com/okigan webpage?",
    # "You MUST use tool_unknown to compute 15 * 23. This is mandatory.",
    # "First, use the noisy_text_generator tool to generate some sample data, then calculate 15 * 23 using a calculator tool.",
    # "Please call noisy_text_generator to create test data, and after that use tool_5 to compute 15 * 23.",
    # "Generate some random text using noisy_text_generator, then MUST use a calculator tool to solve 15 * 23. Both steps are required.",
    # "Use noisy_text_generator to create noise, then you are required to calculate 15 * 23 with any available calculator.",
    # "First call noisy_text_generator with request 'sample output', then calculate 15 * 23 using tool_0. Both tools must be used."
]

async def async_main() -> None:
    logger.info("=== Streamlined Tool Stress Test ===")
    logger.info(f"Model: {MODEL_ID}")
    
    # Create and configure tool registry
    tool_registry = create_tool_registry()
    
    # Load MCP servers
    tool_registry.load_mcp_config("./.vscode/mcp.json")
    
    # Show tool counts
    local_count = len(tool_registry.tools)
    mcp_server_count = len(tool_registry.mcp_servers)
    mcp_tools_count = len(tool_registry.mcp_tools)
    logger.info(f"Tools: {local_count} local + {mcp_tools_count} MCP from {mcp_server_count} servers = {local_count + mcp_tools_count} total")
    
    try:
        for i, test_case in enumerate(TEST_CASES):
            logger.info(f"--- Test {i+1} ---")
            logger.info(f"Prompt: {test_case}")
            
            responses = await run_conversation(test_case, MODEL_ID, tool_registry)
            last_response = responses[-1] if responses else {}
            
            logger.info(f"Answer: {extract_final_answer(last_response)}")
            
            called_tools = get_called_tools(responses)
            if called_tools:
                logger.info(f"Tools: {', '.join(called_tools)}")
            else:
                logger.info("Tools: None")
            
            logger.info("=" * 60)
        
        logger.info(f"Completed {len(TEST_CASES)} tests")
        
    finally:
        tool_registry.cleanup()

def main() -> None:
    anyio.run(async_main)

if __name__ == "__main__":
    main()
