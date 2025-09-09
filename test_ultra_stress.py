#!/usr/bin/env python3
"""
Simplified test to check if models call non-existent tools when overwhelmed.
Provides many tools and checks if the model tries to call tools that don't exist.
"""

import boto3
import random
import string
import inspect
import json
import subprocess
import select
import time

from pathlib import Path
from typing import get_type_hints, get_origin, Optional

MODEL_ID = "openai.gpt-oss-20b-1:0"

class ToolRegistry:
    """Single registry that manages both tool functions and their Bedrock specifications"""
    
    def __init__(self):
        self.tools = {}  # {tool_name: function}
        self.tool_metadata = {}  # {tool_name: {description, param_descriptions, etc.}}
        self.mcp_servers = {}  # {server_name: process}
        self.mcp_tools = {}  # {tool_name: server_name}
    
    def register(self, func, tool_name: Optional[str] = None, **metadata):
        """
        Register a tool function and auto-generate its spec
        
        Args:
            func: The function to register
            tool_name: Optional custom name for the tool
            **metadata: Custom metadata (description, param_descriptions, etc.)
        """
        name = tool_name or func.__name__
        self.tools[name] = func
        if metadata:
            self.tool_metadata[name] = metadata
    
    def register_mcp_server(self, server_name: str, command: list):
        """Register and start an MCP server"""
        try:
            print(f"Starting MCP server '{server_name}' with command: {' '.join(command)}")
            
            # Start MCP server process
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give the process a moment to start
            time.sleep(0.5)
            
            # Check if process started successfully
            if process.poll() is not None:
                try:
                    stderr_output = process.stderr.read() if process.stderr else "No error output"
                except Exception:
                    stderr_output = "Could not read error output"
                print(f"MCP server '{server_name}' failed to start. Error: {stderr_output}")
                return self
            
            self.mcp_servers[server_name] = process
            
            # Get available tools from MCP server
            self._load_mcp_tools(server_name, process)
            
            return self
            
        except Exception as e:
            print(f"Failed to start MCP server '{server_name}': {e}")
            return self
    
    def _load_mcp_tools(self, server_name: str, process):
        """Load tools from MCP server using stdio protocol"""
        try:
            # First, send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "aws-bedrock-tester",
                        "version": "1.0.0"
                    }
                }
            }
            
            print(f"Initializing MCP server '{server_name}'...")
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read initialization response with timeout            
            timeout = 10.0  # 10 seconds timeout for initialization
            start_time = time.time()
            
            # Wait for initialization response
            while time.time() - start_time < timeout:
                if select.select([process.stdout], [], [], 0.1)[0]:
                    response_line = process.stdout.readline()
                    if response_line.strip():
                        init_response = json.loads(response_line)
                        
                        if "result" in init_response:
                            print(f"✓ MCP server '{server_name}' initialized successfully")
                            break
                        elif "error" in init_response:
                            print(f"✗ MCP server '{server_name}' initialization failed: {init_response['error']}")
                            return
                
                if process.poll() is not None:
                    print(f"✗ MCP server '{server_name}' process terminated during initialization")
                    return
            else:
                print(f"✗ Timeout during initialization of MCP server '{server_name}'")
                return
            
            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notification) + "\n")
            process.stdin.flush()
            
            # Now request tools list
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            print(f"Requesting tools from MCP server '{server_name}'...")
            process.stdin.write(json.dumps(tools_request) + "\n")
            process.stdin.flush()
            
            # Read tools response with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                if select.select([process.stdout], [], [], 0.1)[0]:
                    response_line = process.stdout.readline()
                    if response_line.strip():
                        response = json.loads(response_line)
                        
                        if "result" in response and "tools" in response["result"]:
                            tools_loaded = 0
                            print(f"Available tools from '{server_name}':")
                            for tool in response["result"]["tools"]:
                                tool_name = tool["name"]
                                tool_description = tool.get("description", f"MCP tool: {tool_name}")
                                self.mcp_tools[tool_name] = server_name
                                
                                # Store tool metadata for Bedrock spec generation
                                self.tool_metadata[tool_name] = {
                                    "description": tool_description,
                                    "mcp_schema": tool.get("inputSchema", {}),
                                    "is_mcp": True
                                }
                                tools_loaded += 1
                                print(f"  • {tool_name}: {tool_description}")
                            
                            print(f"✓ Loaded {tools_loaded} tools from MCP server '{server_name}'")
                            return
                        elif "error" in response:
                            print(f"✗ MCP server '{server_name}' returned error: {response['error']}")
                            return
                        else:
                            print(f"✗ Unexpected response from MCP server '{server_name}': {response}")
                            return
                
                # Check if process is still alive
                if process.poll() is not None:
                    print(f"✗ MCP server '{server_name}' process terminated unexpectedly")
                    return
            
            print(f"✗ Timeout waiting for tools response from MCP server '{server_name}'")
                    
        except Exception as e:
            print(f"✗ Failed to load tools from MCP server '{server_name}': {e}")
    
    def load_mcp_config(self, config_path: str):
        """Load MCP servers from configuration file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                print(f"MCP config file not found: {config_path}")
                return self
                
            with open(config_file) as f:
                config = json.load(f)
                
            # Load servers from config
            servers = config.get("servers", {})
            if not servers:
                print(f"No servers found in MCP config: {config_path}")
                return self
                
            print(f"Found {len(servers)} MCP servers in config")
            
            for server_name, server_config in servers.items():
                # Build command from command + args
                command = server_config.get("command")
                args = server_config.get("args", [])
                
                if command:
                    # If command is a string, make it a list
                    if isinstance(command, str):
                        full_command = [command] + args
                    else:
                        full_command = command + args
                    
                    self.register_mcp_server(server_name, full_command)
                else:
                    print(f"No command specified for MCP server '{server_name}', skipping")
                    
            return self
            
        except Exception as e:
            print(f"Failed to load MCP config from '{config_path}': {e}")
            print("Continuing without MCP servers...")
            return self
    
    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool by name (either local function or MCP tool)"""
        # Check if it's an MCP tool
        if tool_name in self.mcp_tools:
            return self._execute_mcp_tool(tool_name, tool_input)
        
        # Execute local function
        if tool_name in self.tools:
            func = self.tools[tool_name]
            # Extract parameters based on function signature
            sig = inspect.signature(func)
            kwargs = {}
            for param_name in sig.parameters.keys():
                if param_name in tool_input:
                    kwargs[param_name] = tool_input[param_name]
            return func(**kwargs)
        else:
            return f"Error: Unknown tool {tool_name}"
    
    def _execute_mcp_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute an MCP tool via its server"""
        try:
            server_name = self.mcp_tools[tool_name]
            process = self.mcp_servers[server_name]
            
            # Send tools/call request
            request = {
                "jsonrpc": "2.0",
                "id": 3,  # Use different ID from initialization
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": tool_input
                }
            }
            
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            
            # Read response with timeout
            import select
            import time
            
            timeout = 30.0  # 30 seconds timeout for tool execution
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if select.select([process.stdout], [], [], 0.1)[0]:
                    response_line = process.stdout.readline()
                    if response_line.strip():
                        response = json.loads(response_line)
                        
                        if "result" in response:
                            content = response["result"].get("content", [])
                            if content and len(content) > 0:
                                # Handle different content types
                                first_content = content[0]
                                if isinstance(first_content, dict):
                                    return first_content.get("text", str(first_content))
                                else:
                                    return str(first_content)
                            else:
                                return "No content returned from MCP tool"
                        elif "error" in response:
                            error = response["error"]
                            return f"MCP Error: {error.get('message', 'Unknown error')}"
                        
                # Check if process is still alive
                if process.poll() is not None:
                    return f"MCP server {server_name} terminated unexpectedly"
            
            return f"Timeout executing MCP tool {tool_name}"
                
        except Exception as e:
            return f"Error executing MCP tool {tool_name}: {e}"
    
    def get_bedrock_specs(self) -> list:
        """Get all tools as Bedrock tool specifications"""
        specs = []
        
        # Add local function tools
        for tool_name, func in self.tools.items():
            spec = self._create_spec_from_function(func, tool_name)
            specs.append(spec)
        
        # Add MCP tools
        for tool_name in self.mcp_tools.keys():
            spec = self._create_spec_from_mcp_tool(tool_name)
            specs.append(spec)
            
        return specs
    
    def _create_spec_from_mcp_tool(self, tool_name: str) -> dict:
        """Create a Bedrock tool specification from an MCP tool"""
        metadata = self.tool_metadata.get(tool_name, {})
        mcp_schema = metadata.get('mcp_schema', {})
        
        # Convert MCP schema to Bedrock format
        properties = mcp_schema.get('properties', {})
        required = mcp_schema.get('required', [])
        
        return {
            "toolSpec": {
                "name": tool_name,
                "description": metadata.get('description', f"MCP tool: {tool_name}"),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
        }
    
    def _create_spec_from_function(self, func, tool_name: str) -> dict:
        """Create a Bedrock tool specification from a Python function"""
        metadata = self.tool_metadata.get(tool_name, {})
        
        # Use custom description if provided, otherwise use docstring
        description = metadata.get('description') or func.__doc__ or f"Tool: {tool_name}"
        
        # Get function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        properties = {}
        required = []
        
        # Custom parameter descriptions from metadata
        param_descriptions = metadata.get('param_descriptions', {})
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':  # Skip self parameter
                continue
                
            param_type = type_hints.get(param_name, str)
            
            # Convert Python types to JSON schema types
            if param_type is str:
                json_type = "string"
            elif param_type is int:
                json_type = "integer"
            elif param_type is float:
                json_type = "number"
            elif param_type is bool:
                json_type = "boolean"
            elif get_origin(param_type) is list:
                json_type = "array"
            elif get_origin(param_type) is dict:
                json_type = "object"
            else:
                json_type = "string"  # Default fallback
            
            # Use custom parameter description if provided
            param_desc = param_descriptions.get(param_name, f"Parameter: {param_name}")
            
            properties[param_name] = {
                "type": json_type,
                "description": param_desc
            }
            
            # Check if parameter is required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "toolSpec": {
                "name": tool_name,
                "description": description.strip(),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
        }
    
    def cleanup_mcp_servers(self):
        """Cleanup MCP server processes"""
        for server_name, process in self.mcp_servers.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"Stopped MCP server: {server_name}")
            except Exception as e:
                print(f"Error stopping MCP server {server_name}: {e}")
        self.mcp_servers.clear()
        self.mcp_tools.clear()
    
    def list_tools_by_type(self):
        """List all tools organized by type (local vs MCP servers)"""
        print("\n=== Available Tools ===")
        
        # List local tools
        local_tools = list(self.tools.keys())
        if local_tools:
            print(f"Local Python functions ({len(local_tools)}):")
            for tool_name in sorted(local_tools):
                description = self.tool_metadata.get(tool_name, {}).get('description', 'No description')
                print(f"  • {tool_name}: {description}")
        else:
            print("Local Python functions: None")
        
        # List MCP tools by server
        if self.mcp_tools:
            print(f"\nMCP Tools ({len(self.mcp_tools)}):")
            
            # Group tools by server
            tools_by_server = {}
            for tool_name, server_name in self.mcp_tools.items():
                if server_name not in tools_by_server:
                    tools_by_server[server_name] = []
                tools_by_server[server_name].append(tool_name)
            
            for server_name, tool_names in tools_by_server.items():
                print(f"  {server_name} ({len(tool_names)} tools):")
                for tool_name in sorted(tool_names):
                    description = self.tool_metadata.get(tool_name, {}).get('description', 'No description')
                    print(f"    • {tool_name}: {description}")
        else:
            print("\nMCP Tools: None")
        
        total_tools = len(self.tools) + len(self.mcp_tools)
        print(f"\nTotal tools available: {total_tools}")
        print("=" * 50)
    
    def get_tool_details(self, tool_name: str) -> dict:
        """Get detailed information about a specific tool"""
        if tool_name in self.tools:
            func = self.tools[tool_name]
            return {
                "type": "local_function",
                "name": tool_name,
                "function": func.__name__,
                "docstring": func.__doc__,
                "metadata": self.tool_metadata.get(tool_name, {}),
                "signature": str(inspect.signature(func))
            }
        elif tool_name in self.mcp_tools:
            server_name = self.mcp_tools[tool_name]
            metadata = self.tool_metadata.get(tool_name, {})
            return {
                "type": "mcp_tool",
                "name": tool_name,
                "server": server_name,
                "description": metadata.get("description", "No description"),
                "schema": metadata.get("mcp_schema", {}),
                "metadata": metadata
            }
        else:
            return {"error": f"Tool '{tool_name}' not found"}

# Create the registry
tool_registry = ToolRegistry()

# Tool functions with proper docstrings
def calculator_tool(expression: str) -> str:
    """
    Calculator tool - performs mathematical calculations and arithmetic operations.
    
    Args:
        expression: Mathematical expression to calculate (e.g., '15 * 23')
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Safely evaluate mathematical expressions
        allowed_chars = set('0123456789+-*/().')
        if all(c in allowed_chars or c.isspace() for c in expression):
            result = str(eval(expression))
            return result
        else:
            return "Error: Invalid expression"
    except Exception:
        return "Error: Could not evaluate expression"

def noisy_text_generator(request: str = "generate text") -> str:
    """
    Generates random text and noise - useful for creating sample data or testing.
    
    Args:
        request: What kind of text or data to generate
    
    Returns:
        Generated noisy text with extraneous characters
    """
    # Generate noisy output with various extraneous characters
    noise_chars = "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
    random_noise = ''.join(random.choices(
        noise_chars + string.ascii_letters + string.digits + ' \n\t', 
        k=200
    ))
    
    # Mix in some actual response
    actual_response = f"Generated text based on request: {request}"
    
    # Create noisy output
    return f"""
NOISE_START_{random_noise[:50]}_NOISE_MID
{actual_response}
EXTRA_CHARS: ¡™£¢∞§¶•ªº–≠œ∑´®†¥¨ˆøπ"'«åß∂ƒ©˙∆˚¬…æΩ≈ç√∫˜µ≤≥÷
MORE_NOISE_{random_noise[50:100]}_END_NOISE
RANDOM_SYMBOLS: ◊Ω≈ç√∫˜µ≤≥÷æ…¬Ω≈ç√∫˜
{random_noise[100:150]}
FINAL_NOISE_BLOCK_{random_noise[150:]}_COMPLETE
"""

# Register tools
NUM_TOOLS = 50

# Register calculator tools (multiple instances of the same tool)
for i in range(NUM_TOOLS):
    tool_registry.register(
        calculator_tool, 
        tool_name=f"tool_{i}",
        description=f"Calculator tool {i} - performs mathematical calculations and arithmetic operations"
    )

# Register noisy tool with custom parameter descriptions
tool_registry.register(
    noisy_text_generator,
    param_descriptions={
        'request': 'Specific type of content to generate (e.g., "test data", "random text", "sample output")'
    }
)

# Get Bedrock tool specifications
AVAILABLE_TOOLS = tool_registry.get_bedrock_specs()

def _execute_tool_calls(content) -> list:
    """Extract and execute tool calls from assistant message content"""
    tool_results = []
    for item in content:
        if isinstance(item, dict) and 'toolUse' in item:
            tool_use = item['toolUse']
            tool_id = tool_use.get('toolUseId', 'unknown')
            tool_name = tool_use.get('name', 'unknown')
            tool_input = tool_use.get('input', {})
            
            # Execute tool using registry
            result = tool_registry.execute(tool_name, tool_input)
            
            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_id,
                    "content": [{"text": result}]
                }
            })
    return tool_results

def _has_tool_calls(content) -> bool:
    """Check if message content contains tool calls"""
    return any(
        isinstance(item, dict) and 'toolUse' in item 
        for item in content if isinstance(content, list)
    )

def run_conversation_with_tools(prompt) -> list:
    """Run a complete conversation with tool execution until completion"""
    bedrock = boto3.client("bedrock-runtime")
    
    try:
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        all_responses = []
        max_turns = 5  # Prevent infinite loops
        
        for turn in range(max_turns):
            # Get response from Bedrock
            response = bedrock.converse(
                modelId=MODEL_ID,
                messages=messages,
                toolConfig={
                    "tools": AVAILABLE_TOOLS,
                    "toolChoice": {"any": {}}
                }
            )
            all_responses.append(response)
            
            # Handle response based on stop reason
            stop_reason = response.get('stopReason')
            if not stop_reason:
                print(f"No valid response, breaking conversation: {response}")
                break
                
            if stop_reason == 'tool_use':
                # Process tool calls
                assistant_message = response.get('output', {}).get('message')
                if not assistant_message:
                    break
                    
                messages.append(assistant_message)
                content = assistant_message.get('content', [])
                
                if _has_tool_calls(content):
                    tool_results = _execute_tool_calls(content)
                    if tool_results:
                        messages.append({
                            "role": "user",
                            "content": tool_results
                        })
                else:
                    break  # No more tool calls
                    
            elif stop_reason in ['end_turn', 'max_turns', 'model_complete']:
                print(f"Conversation ended due to {stop_reason}")
                break
            else:
                print(f"Unknown stop reason, breaking conversation: {stop_reason}")
                break

        return all_responses
        
    except Exception as e:
        return [{"error": str(e)}]


def extract_final_answer(response):
    """Extract the final text answer from the response"""
    if isinstance(response, dict) and 'output' in response:
        output = response['output']
        if isinstance(output, dict) and 'message' in output:
            message = output['message']
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                
                # Look for text content (not tool calls)
                for item in content if isinstance(content, list) else []:
                    if isinstance(item, dict) and 'text' in item:
                        return item['text']
    
    return "No final answer found"

def get_all_called_tools(responses):
    """Extract all tool names that were called in the responses"""
    called_tools = []

    for response in responses:
        if isinstance(response, dict) and 'output' in response:
            output = response['output']
            if isinstance(output, dict) and 'message' in output:
                message = output['message']
                if isinstance(message, dict) and 'content' in message:
                    content = message['content']
                    for item in content if isinstance(content, list) else []:
                        if isinstance(item, dict) and 'toolUse' in item:
                            tool_use = item['toolUse']
                            if 'name' in tool_use:
                                called_tools.append(tool_use['name'])

    return called_tools

# Test cases designed to trigger calls to non-existent tools
TEST_CASES = [
    # # Force tool usage with explicit requirement
    # "You MUST use a calculator tool to compute 15 * 23. Do not calculate manually.",
    
    # # Request for specific numbered tools (some may not exist in prompts)
    # "You are required to use tool_0 to solve this problem: 15 * 23",
    
    # # Request that might confuse the model about available tools
    # "You MUST use tool_100 to calculate 15 * 23. This is mandatory.",
    
    # # Force tool usage with explicit requirement
    # "Use any calculator tool available to compute 15 * 23. Tool usage is required.",
    
    # # Request for tools with similar names - force usage
    # "You MUST use calc_tool to solve 15 * 23. Tool usage is mandatory.",
    
    # # Force usage of specific tools
    # "You are required to use tool_0 to calculate 15 * 23",
    
    # # Request for non-existent tool patterns with force
    # "You MUST use tool_calculate to solve 15 * 23. This is required.",
    
    # Mixed existing and non-existing tools with force
    "You MUST use tool_unknown to compute 15 * 23. This is mandatory.",
    
    # Test cases with noisy tool followed by calculation request
    "First, use the noisy_text_generator tool to generate some sample data, then calculate 15 * 23 using a calculator tool.",
    
    "Please call noisy_text_generator to create test data, and after that use tool_5 to compute 15 * 23.",
    
    "Generate some random text using noisy_text_generator, then MUST use a calculator tool to solve 15 * 23. Both steps are required.",
    
    "Use noisy_text_generator to create noise, then you are required to calculate 15 * 23 with any available calculator.",
    
    "First call noisy_text_generator with request 'sample output', then calculate 15 * 23 using tool_0. Both tools must be used."
]

def main():
    print("=== Test for Non-Existent Tool Calls ===")
    print(f"Model: {MODEL_ID}")
    
    # Load MCP servers if config exists
    mcp_config_path = "./.vscode/mcp.json"
    if Path(mcp_config_path).exists():
        print(f"Loading MCP servers from {mcp_config_path}...")
        tool_registry.load_mcp_config(mcp_config_path)
    else:
        print("No MCP config found, using only local tools")
        # Example: manually add an MCP server (commented out)
        # tool_registry.register_mcp_server("filesystem", ["npx", "@modelcontextprotocol/server-filesystem", "/tmp"])
    
    # List all available tools
    tool_registry.list_tools_by_type()
    
    # Show available tools (legacy format for backward compatibility)
    all_tools = list(tool_registry.tools.keys()) + list(tool_registry.mcp_tools.keys())
    print(f"\nRunning tests with {len(all_tools)} total tools available")
    print()
    
    try:        
        for i, test_case in enumerate(TEST_CASES):
            print(f"--- Test {i+1} ---")
            print(f"Prompt: {test_case}")
            print()
            
            responses = run_conversation_with_tools(test_case)
            last_response = responses[-1] if responses else {"error": "No responses received"}

            # print(f"Response: {last_response}")
            
            # Extract and show the final answer
            final_answer = extract_final_answer(last_response)
            print(f"Final Answer: {final_answer}")
            
            # Show what tools were called
            called_tools = get_all_called_tools(responses)
            if called_tools:
                print(f"Tools called: {', '.join(called_tools)}")
            else:
                print("No tools were called")
            
            
            print("\n" + "="*60 + "\n")
        
        print("=== Final Summary ===")
        print(f"Total tests: {len(TEST_CASES)}")
        print(f"Total available tools: {len(all_tools)}")
        
    finally:
        # Cleanup MCP servers
        if tool_registry.mcp_servers:
            print("Cleaning up MCP servers...")
            tool_registry.cleanup_mcp_servers()

if __name__ == "__main__":
    main()
