#!/usr/bin/env python3
"""
Simplified test to check if models call non-existent tools when overwhelmed.
Provides many tools and checks if the model tries to call tools that don't exist.
"""

import boto3

MODEL_ID = "openai.gpt-oss-20b-1:0"

# Create many tools to overwhelm the model
NUM_TOOLS = 50
AVAILABLE_TOOLS = []

for i in range(NUM_TOOLS):
    tool = {
        "toolSpec": {
            "name": f"tool_{i}",
            "description": f"Calculator tool {i} - performs mathematical calculations and arithmetic operations",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to calculate (e.g., '15 * 23')"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    }
    AVAILABLE_TOOLS.append(tool)

def execute_tool(tool_name, tool_input):
    """Actually execute the specified tool"""
    # Extract the expression from input
    expression = tool_input.get('expression', '')
    
    # Simple calculator implementation
    try:
        # Safely evaluate mathematical expressions
        # Only allow basic math operations
        allowed_chars = set('0123456789+-*/().')
        if all(c in allowed_chars or c.isspace() for c in expression):
            result = str(eval(expression))
            return result
        else:
            return "Error: Invalid expression"
    except Exception:
        return "Error: Could not evaluate expression"

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
                                    
                                    # Actually execute the tool
                                    result = execute_tool(tool_name, tool_input)
                                    
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
    # Force tool usage with explicit requirement
    "You MUST use a calculator tool to compute 15 * 23. Do not calculate manually.",
    
    # Request for specific numbered tools (some may not exist in prompts)
    "You are required to use tool_0 to solve this problem: 15 * 23",
    
    # Request that might confuse the model about available tools
    "You MUST use tool_100 to calculate 15 * 23. This is mandatory.",
    
    # Force tool usage with explicit requirement
    "Use any calculator tool available to compute 15 * 23. Tool usage is required.",
    
    # Request for tools with similar names - force usage
    "You MUST use calc_tool to solve 15 * 23. Tool usage is mandatory.",
    
    # Force usage of specific tools
    "You are required to use tool_0 to calculate 15 * 23",
    
    # Request for non-existent tool patterns with force
    "You MUST use tool_calculate to solve 15 * 23. This is required.",
    
    # Mixed existing and non-existing tools with force
    "You MUST use tool_unknown to compute 15 * 23. This is mandatory."
]

def main():
    print("=== Test for Non-Existent Tool Calls ===")
    print(f"Model: {MODEL_ID}")
    print(f"Available tools: tool_0 through tool_{NUM_TOOLS-1}")
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

if __name__ == "__main__":
    main()
