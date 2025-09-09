
import boto3

# Model IDs for AWS Bedrock Converse API
MODELS = {
    # "llama3": "us.meta.llama3-1-8b-instruct-v1:0",
    "gpt_oss": "openai.gpt-oss-20b-1:0"
    # "gpt_oss": "mistral.gpt-oss-20b-v1"
}

# Tool definitions with schemas
TOOL_DEFINITIONS = {
    "calculator": {
        "function": lambda expression: eval(expression, {"__builtins__": {}}, {}),
        "schema": {
            "name": "calculator",
            "description": "Performs basic arithmetic operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate."}
                },
                "required": ["expression"]
            }
        }
    },
    "get_weather": {
        "function": lambda city, country=None: f"The weather in {city}{', ' + country if country else ''} is sunny, 22°C with light clouds.",
        "schema": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "country": {"type": "string", "description": "Country code (optional)"}
                },
                "required": ["city"]
            }
        }
    }
}

# Test cases using tool names instead of duplicating schemas
TEST_CASES = [
    {
        "name": "simple_question",
        "prompt": "What is the capital of France?",
        "tools": None
    },
    {
        "name": "math_with_tools", 
        "prompt": "Calculate 15 * 23 + 7. Use the calculator tool if available.",
        "tools": ["calculator"]
    },
    {
        "name": "weather_with_tools",
        "prompt": "What's the weather like in Paris? Use weather tool if available.",
        "tools": ["get_weather"]
    }
]

bedrock = boto3.client("bedrock-runtime")

def execute_tool(tool_name, tool_input):
    """Execute the requested tool and return the result"""
    if tool_name not in TOOL_DEFINITIONS:
        return f"Tool '{tool_name}' is not implemented."
    
    try:
        tool_function = TOOL_DEFINITIONS[tool_name]["function"]
        return f"The result is: {tool_function(**tool_input)}"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

def build_tool_config(tool_names):
    """Build tool configuration from tool names"""
    if not tool_names:
        return None
    
    config = {
        "tools": [
            {
                "toolSpec": {
                    "name": TOOL_DEFINITIONS[name]["schema"]["name"],
                    "description": TOOL_DEFINITIONS[name]["schema"]["description"],
                    "inputSchema": {"json": TOOL_DEFINITIONS[name]["schema"]["parameters"]}
                }
            } for name in tool_names
        ],
        "toolChoice": {"auto": {}}
    }
    
    return config

def query_converse_with_tools(model_id, initial_prompt, tool_names=None, max_iterations=5):
    """Run a conversation with tool support until completion"""
    messages = [{"role": "user", "content": [{"text": initial_prompt}]}]
    
    for iteration in range(max_iterations):
        kwargs = {"modelId": model_id, "messages": messages}
        
        tool_config = build_tool_config(tool_names)
        if tool_config:
            kwargs["toolConfig"] = tool_config
        
        response = bedrock.converse(**kwargs)
        
        # Add assistant's response to conversation
        assistant_message = response["output"]["message"]
        messages.append(assistant_message)
        
        # Check if assistant wants to use tools
        tool_calls_made = False
        for content_block in assistant_message.get("content", []):
            if "toolUse" in content_block:
                tool_calls_made = True
                tool_use = content_block["toolUse"]
                tool_result = execute_tool(tool_use["name"], tool_use.get("input", {}))
                
                # Add tool result to conversation
                messages.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"text": tool_result}]
                        }
                    }]
                })
        
        # If no tools were called, conversation is complete
        if not tool_calls_made:
            return response, messages
    
    return response, messages

def process_conversation_output(final_response, all_messages):
    """Process the entire conversation including tool calls"""
    print("=== Full Conversation ===")
    
    for message in all_messages:
        role = message.get("role", "unknown")
        
        for content_block in message.get("content", []):
            if "text" in content_block:
                print(f"{role.title()}: {content_block['text']}")
            elif "toolResult" in content_block:
                tool_content = content_block["toolResult"]["content"][0]["text"]
                print(f"Tool Result: {tool_content}")
            elif "toolUse" in content_block:
                tool_use = content_block["toolUse"]
                print(f"Tool Call: {tool_use['name']}({tool_use.get('input', {})})")
    
    # Show usage metrics
    if 'usage' in final_response:
        usage = final_response['usage']
        print(f"\nTokens - Input: {usage.get('inputTokens', 0)}, Output: {usage.get('outputTokens', 0)}, Total: {usage.get('totalTokens', 0)}")
    
    print("=== End Conversation ===\n")

def main():
    for name, model_id in MODELS.items():
        print(f"\n=== Testing Model: {name} ({model_id}) ===")
        
        for test_case in TEST_CASES:
            print(f"\n--- Test: {test_case['name']} ---")
            print(f"Initial Prompt: {test_case['prompt']}")
            
            try:
                final_response, all_messages = query_converse_with_tools(
                    model_id, 
                    test_case['prompt'], 
                    test_case['tools']
                )
                
                process_conversation_output(final_response, all_messages)
                    
            except Exception as e:
                print(f"Error with {name} on {test_case['name']}: {e}")

if __name__ == "__main__":
    main()
