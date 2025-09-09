#!/usr/bin/env python3
"""
Simplified test to check if models call non-existent tools when overwhelmed.
Provides many tools and checks if the model tries to call tools that don't exist.
"""

import boto3
import random
import string
import inspect
from typing import get_type_hints, get_origin, Optional

MODEL_ID = "openai.gpt-oss-20b-1:0"

class ToolRegistry:
    """Single registry that manages both tool functions and their Bedrock specifications"""
    
    def __init__(self):
        self.tools = {}  # {tool_name: function}
        self.tool_descriptions = {}  # {tool_name: custom_description}
    
    def register(self, func, tool_name: Optional[str] = None, description: Optional[str] = None):
        """Register a tool function and auto-generate its spec"""
        name = tool_name or func.__name__
        self.tools[name] = func
        if description:
            self.tool_descriptions[name] = description
    
    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool by name"""
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
    
    def get_bedrock_specs(self) -> list:
        """Get all tools as Bedrock tool specifications"""
        specs = []
        for tool_name, func in self.tools.items():
            spec = self._create_spec_from_function(func, tool_name)
            specs.append(spec)
        return specs
    
    def _create_spec_from_function(self, func, tool_name: str) -> dict:
        """Create a Bedrock tool specification from a Python function"""
        # Use custom description if provided, otherwise use docstring
        if tool_name in self.tool_descriptions:
            description = self.tool_descriptions[tool_name]
        else:
            description = func.__doc__ or f"Tool: {tool_name}"
        
        # Get function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        properties = {}
        required = []
        
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
            
            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter: {param_name}"
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
        f"tool_{i}",
        f"Calculator tool {i} - performs mathematical calculations and arithmetic operations"
    )

# Register noisy tool
tool_registry.register(noisy_text_generator)

# Get Bedrock tool specifications
AVAILABLE_TOOLS = tool_registry.get_bedrock_specs()

def test_with_many_tools(prompt) -> list:
    """Test with many tools and handle full conversation flow with actual tool execution"""
    bedrock = boto3.client("bedrock-runtime")
    
    try:
        # Initial conversation
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        # Continue conversation until no more tool calls
        max_turns = 5  # Prevent infinite loops
        all_responses = []
        
        for turn in range(max_turns):
            response = bedrock.converse(
                modelId=MODEL_ID,
                messages=messages,
                toolConfig={
                    "tools": AVAILABLE_TOOLS,
                    "toolChoice": {"any": {}}
                }
            )
            
            all_responses.append(response)
            
            # Add assistant's response to conversation
            if 'stopReason' in response:
                if response['stopReason'] == 'tool_use':
                    if 'output' in response and 'message' in response['output']:
                        assistant_message = response['output']['message']
                        messages.append(assistant_message)
                        
                        # Check if there are tool calls
                        content = assistant_message.get('content', [])
                        has_tool_calls = any(
                            isinstance(item, dict) and 'toolUse' in item 
                            for item in content if isinstance(content, list)
                        )
                        
                        if has_tool_calls:
                            # Execute actual tools
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
                            
                            # Add tool results to conversation
                            if tool_results:
                                messages.append({
                                    "role": "user",
                                    "content": tool_results
                                })
                        else:
                            # No more tool calls, conversation is complete
                            break
                    else:
                        # No valid response, break
                        break
                elif response['stopReason'] in ['end_turn', 'max_turns', 'model_complete']:
                    print(f"Conversation ended due to {response['stopReason']}")
                    break
                else:
                    print(f"No valid stopReason, breaking conversation: {response}")
                    break
            else:
                print(f"No valid response, breaking conversation: {response}")
                break

        # Return the last response for analysis
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
    print(f"Available tools: tool_0 through tool_{NUM_TOOLS-1}, noisy_text_generator")
    print()
        
    for i, test_case in enumerate(TEST_CASES):
        print(f"--- Test {i+1} ---")
        print(f"Prompt: {test_case}")
        print()
        
        responses = test_with_many_tools(test_case)
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
    print(f"Available tools: tool_0 through tool_{NUM_TOOLS-1}, noisy_text_generator")

if __name__ == "__main__":
    main()
