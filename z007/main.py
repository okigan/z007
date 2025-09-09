#!/usr/bin/env python3
"""
z007 - Fast Micro Agent Interactive REPL
"""

import json
import anyio
import colorlog
import sys

from pathlib import Path
from typing import Any, Callable
from dotenv import load_dotenv
from .agent import Agent

# Load environment variables from .env file
load_dotenv()

# Set up colored logging
colorlog.basicConfig(format='%(log_color)s%(asctime)s - %(levelname)s - %(message)s')
logger = colorlog.getLogger(__name__)


def get_called_tools(responses: list[Any]) -> list[str]:
    """Get list of called tools"""
    tools: list[str] = []
    for response in responses:
        try:
            content = response.get('output', {}).get('message', {}).get('content', [])
            for item in content:
                if isinstance(item, dict) and 'toolUse' in item:
                    tool_use = item['toolUse']
                    if tool_use:
                        tool_name = tool_use.get('name')
                        if tool_name:
                            tools.append(str(tool_name))
        except Exception:
            continue
    return tools


def create_tools() -> list[Callable[..., Any]]:
    """Create and return list of tool functions"""
    tools = []
    
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

    # Add basic tools
    tools.extend([calculator_tool])
    
    return tools


def load_mcp_config_from_file(config_path: str) -> dict[str, Any] | None:
    """Load MCP configuration from JSON file"""
    try:
        if not Path(config_path).exists():
            return None
        
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading MCP config from {config_path}: {e}")
        return None


def print_help():
    """Print help information"""
    print("\n=== z007 - Fast Micro Agent ===")
    print("Commands:")
    print("  /help     - Show this help message")
    print("  /tools    - List available tools")
    print("  /clear    - Clear conversation history")
    print("  /exit     - Exit the REPL")
    print("\nJust type your message to chat with the AI assistant.")
    print("The assistant has access to various tools including calculators and MCP tools.\n")


def print_tools_info(agent: Agent):
    """Print information about available tools"""
    local_count, mcp_server_count, mcp_tools_count = agent.get_tool_counts()
    print("\n=== Tool Information ===")
    print(f"Local tools: {local_count}")
    print(f"MCP servers: {mcp_server_count}")
    print(f"MCP tools: {mcp_tools_count}")
    print(f"Total tools: {local_count + mcp_tools_count}")
    
    # List some tool names
    local_tools, mcp_tools = agent.get_tool_names()
    if local_tools:
        print(f"\nLocal tools: {local_tools}")
    if mcp_tools:
        print(f"MCP tools: {mcp_tools}")
    print()


async def async_main() -> None:
    """Main REPL function"""
    model_id = "openai.gpt-oss-20b-1:0"
    mcp_config_filepath = "./.vscode/mcp.json"

    print("Starting z007 - Fast Micro Agent...")
    logger.info(f"Model: {model_id}")

    # Create tools and agent
    tools = create_tools()

    try:
        async with Agent(
            model_id=model_id,
            system_prompt="You are a helpful assistant with access to various tools. Be concise but informative in your responses.",
            tools=tools,
            mcp_config=load_mcp_config_from_file(mcp_config_filepath) if Path(mcp_config_filepath).exists() else None,
            max_turns=10  # Allow more turns for interactive conversation
        ) as agent:
            
            # Show initial info
            local_count, mcp_server_count, mcp_tools_count = agent.get_tool_counts()
            print(f"Tools: {local_count} local + {mcp_tools_count} MCP from {mcp_server_count} servers = {local_count + mcp_tools_count} total")
            
            print_help()
            
            # Main REPL loop
            conversation_history = []
            
            while True:
                try:
                    # Get user input
                    user_input = input(">>> ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        command = user_input.lower()
                        
                        if command in ['/exit']:
                            print("Goodbye!")
                            break
                        elif command == '/help':
                            print_help()
                            continue
                        elif command == '/tools':
                            print_tools_info(agent)
                            continue
                        elif command == '/clear':
                            conversation_history.clear()
                            print("Conversation history cleared.")
                            continue
                        else:
                            print(f"Unknown command: {user_input}. Type /help for available commands.")
                            continue
                    
                    # Process user message
                    print("Thinking...")
                    
                    # Run conversation
                    responses = await agent.run_conversation(user_input)
                    last_response = responses[-1] if responses else None
                    
                    # Extract and display answer
                    answer = Agent.extract_final_answer(last_response) if last_response else 'No response'
                    print(f"\nAssistant: {answer}")
                    
                    # Show tools used (if any)
                    tools_used = get_called_tools(responses)
                    if tools_used:
                        print(f"Tools used: {', '.join(tools_used)}")
                    
                    print()  # Add spacing
                    
                    # Store in conversation history for reference
                    conversation_history.append({
                        'user': user_input,
                        'assistant': answer,
                        'tools': tools_used
                    })
                    
                except KeyboardInterrupt:
                    print("\n\nUse /exit to leave the REPL.")
                    continue
                except EOFError:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error processing input: {e}")
                    print(f"Error: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        print(f"Failed to start agent: {e}")
        sys.exit(1)


def main() -> None:
    """Entry point"""
    anyio.run(async_main)


if __name__ == "__main__":
    main()
