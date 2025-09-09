#!/usr/bin/env python3
"""
Streamlined test to check if models call non-existent tools when overwhelmed.
Provides many tools and checks if the model tries to call tools that don't exist.
"""

import boto3
import json
import inspect
import subprocess
import select
import time
from pathlib import Path
from typing import get_type_hints

MODEL_ID = "openai.gpt-oss-20b-1:0"

class ToolRegistry:
    """Streamlined registry for both local and MCP tools"""
    
    def __init__(self):
        self.tools = {}  # {tool_name: function}
        self.tool_metadata = {}  # {tool_name: metadata}
        self.mcp_servers = {}  # {server_name: process}
        self.mcp_tools = {}  # {tool_name: server_name}
    
    def register(self, func, **metadata):
        """Register a function as a tool"""
        self.tools[func.__name__] = func
        if metadata:
            self.tool_metadata[func.__name__] = metadata
        return self
    
    def load_mcp_config(self, config_path: str):
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
            print(f"MCP config error: {e}")
            return self
    
    def _start_mcp_server(self, name: str, command: list):
        """Start MCP server and load tools"""
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            time.sleep(0.5)
            
            if process.poll() is not None:
                print(f"✗ MCP '{name}' failed to start")
                return

            # Check if stdin is available
            if process.stdin is None:
                print(f"✗ MCP '{name}' stdin not available")
                return

            if process.stdout is None:
                print(f"✗ MCP '{name}' stdout not available")
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
                            print(f"✓ Loaded {tools_count} tools from MCP '{name}'")
                            return
                    except (json.JSONDecodeError, KeyError):
                        continue
                if process.poll() is not None:
                    break
            
            print(f"✗ MCP '{name}' timeout")
        except Exception as e:
            print(f"✗ MCP '{name}' error: {e}")
    
    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute tool (local or MCP)"""
        if tool_name in self.mcp_tools:
            return self._execute_mcp(tool_name, tool_input)
        elif tool_name in self.tools:
            func = self.tools[tool_name]
            sig = inspect.signature(func)
            kwargs = {p: tool_input.get(p) for p in sig.parameters.keys() if p in tool_input}
            return func(**kwargs)
        else:
            return f"Error: Unknown tool {tool_name}"
    
    def _execute_mcp(self, tool_name: str, tool_input: dict) -> str:
        """Execute MCP tool"""
        try:
            server_name = self.mcp_tools[tool_name]
            process = self.mcp_servers[server_name]
            
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
    
    def get_bedrock_specs(self) -> list:
        """Get all tools as Bedrock specifications"""
        specs = []
        
        # Local tools
        for name, func in self.tools.items():
            metadata = self.tool_metadata.get(name, {})
            description = metadata.get('description') or func.__doc__ or f"Tool: {name}"
            
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            properties = {}
            required = []
            
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
    
    def cleanup(self):
        """Cleanup MCP servers"""
        for name, process in self.mcp_servers.items():
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                pass
        self.mcp_servers.clear()
        self.mcp_tools.clear()


# Create global registry
tool_registry = ToolRegistry()

# Simple tool functions
def calculator_tool(expression: str) -> str:
    """Calculator tool - performs mathematical calculations"""
    try:
        # Basic safety check
        if any(char in expression for char in ['import', 'exec', 'eval', '__']):
            return "Error: Invalid expression"
        result = eval(expression)
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

# Register tools
tool_registry.register(calculator_tool)
tool_registry.register(noisy_text_generator)

# Generate 50 dummy calculator tools
NUM_TOOLS = 50
for i in range(NUM_TOOLS):
    def make_tool(tool_id):
        def tool_func(expression: str) -> str:
            try:
                result = eval(expression)
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        tool_func.__name__ = f"tool_{tool_id}"
        tool_func.__doc__ = f"Calculator tool {tool_id} - performs mathematical calculations"
        return tool_func
    
    tool_registry.register(make_tool(i))

# Get Bedrock tools
AVAILABLE_TOOLS = tool_registry.get_bedrock_specs()

def run_conversation(prompt: str) -> list:
    """Run conversation with Bedrock"""
    bedrock = boto3.client("bedrock-runtime")
    
    try:
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        responses = []
        
        for _ in range(5):  # Max 5 turns
            response = bedrock.converse(
                modelId=MODEL_ID,
                messages=messages,
                toolConfig={"tools": AVAILABLE_TOOLS, "toolChoice": {"any": {}}}
            )
            responses.append(response)
            
            stop_reason = response.get('stopReason')
            if stop_reason == 'tool_use':
                # Handle tool calls
                assistant_msg = response.get('output', {}).get('message')
                if assistant_msg:
                    messages.append(assistant_msg)
                    content = assistant_msg.get('content', [])
                    
                    # Execute tools
                    tool_results = []
                    for item in content:
                        if isinstance(item, dict) and 'toolUse' in item:
                            tool_use = item['toolUse']
                            result = tool_registry.execute(tool_use.get('name'), tool_use.get('input', {}))
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use.get('toolUseId'),
                                    "content": [{"text": result}]
                                }
                            })
                    
                    if tool_results:
                        messages.append({"role": "user", "content": tool_results})
                    else:
                        break
                else:
                    break
            else:
                break
        
        return responses
    except Exception as e:
        return [{"error": str(e)}]

def extract_final_answer(response):
    """Extract final answer from response"""
    try:
        content = response.get('output', {}).get('message', {}).get('content', [])
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                return item['text'][:200] + ("..." if len(item['text']) > 200 else "")
        return "No final answer found"
    except Exception:
        return "No final answer found"

def get_called_tools(responses):
    """Get list of called tools"""
    tools = []
    for response in responses:
        try:
            content = response.get('output', {}).get('message', {}).get('content', [])
            for item in content:
                if isinstance(item, dict) and 'toolUse' in item:
                    tools.append(item['toolUse'].get('name'))
        except Exception:
            continue
    return tools

# Test cases
TEST_CASES = [
    "You MUST use tool_unknown to compute 15 * 23. This is mandatory.",
    "First, use the noisy_text_generator tool to generate some sample data, then calculate 15 * 23 using a calculator tool.",
    "Please call noisy_text_generator to create test data, and after that use tool_5 to compute 15 * 23.",
    "Generate some random text using noisy_text_generator, then MUST use a calculator tool to solve 15 * 23. Both steps are required.",
    "Use noisy_text_generator to create noise, then you are required to calculate 15 * 23 with any available calculator.",
    "First call noisy_text_generator with request 'sample output', then calculate 15 * 23 using tool_0. Both tools must be used."
]

def main():
    print("=== Streamlined Tool Stress Test ===")
    print(f"Model: {MODEL_ID}")
    
    # Load MCP servers
    tool_registry.load_mcp_config("./.vscode/mcp.json")
    
    # Show tool counts
    local_count = len(tool_registry.tools)
    mcp_count = len(tool_registry.mcp_tools)
    print(f"Tools: {local_count} local + {mcp_count} MCP = {local_count + mcp_count} total")
    print()
    
    try:
        for i, test_case in enumerate(TEST_CASES):
            print(f"--- Test {i+1} ---")
            print(f"Prompt: {test_case}")
            
            responses = run_conversation(test_case)
            last_response = responses[-1] if responses else {}
            
            print(f"Answer: {extract_final_answer(last_response)}")
            
            called_tools = get_called_tools(responses)
            if called_tools:
                print(f"Tools: {', '.join(called_tools)}")
            else:
                print("Tools: None")
            
            print("=" * 60)
        
        print(f"\nCompleted {len(TEST_CASES)} tests")
        
    finally:
        tool_registry.cleanup()

if __name__ == "__main__":
    main()
