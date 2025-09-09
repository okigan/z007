#!/usr/bin/env python3
"""
z007 - Fast Micro Agent Interactive REPL
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import anyio
from dotenv import load_dotenv
from rich.console import Console
from rich import print
from rich.logging import RichHandler

from .agent import Agent

# Set up rich logging with colorization and timestamps on every line
rich_handler = RichHandler(
    console=Console(file=sys.stdout, force_terminal=True),
    show_time=False,
    show_level=True,
    show_path=False,
    rich_tracebacks=True,
    markup=True,
    omit_repeated_times=False  # This forces timestamp on every line
)
rich_handler.setFormatter(logging.Formatter(
    fmt="%(message)s",
    datefmt="[%X]"
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[rich_handler],
    force=True
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[RichHandler(console=Console(file=sys.stdout))]
)
logger = logging.getLogger(__name__)
console = Console()


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


def find_mcp_config_file() -> str | None:
    """Find the first existing MCP config file from possible locations"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    possible_paths = [
        "./mcp.json",
        "./.vscode/mcp.json",
        "~/mcp.json",
        str(script_dir / "mcp.json"),  # Packaged mcp.json in the same directory as this script
    ]
    
    for path in possible_paths:
        expanded_path = Path(path).expanduser()
        if expanded_path.exists():
            return str(expanded_path)
    
    return None


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
    print("[bold cyan]\n=== z007 - Fast Micro Agent ===[/bold cyan]")
    print("[bold yellow]Commands:[/bold yellow]")
    print("  [green]/help[/green]     - Show this help message")
    print("  [green]/tools[/green]    - List available tools")
    print("  [green]/clear[/green]    - Clear conversation history")
    print("  [green]/exit[/green]     - Exit the REPL")
    print("\nJust type your message to chat with the AI assistant.")
    print("The assistant has access to various tools including calculators and MCP tools.\n")


def print_tools_info(agent: Agent):
    """Print information about available tools"""
    local_count, mcp_server_count, mcp_tools_count = agent.get_tool_counts()
    print("[bold magenta]\n=== Tool Information ===[/bold magenta]")
    print(f"[cyan]Local tools:[/cyan] {local_count}")
    print(f"[cyan]MCP servers:[/cyan] {mcp_server_count}")
    print(f"[cyan]MCP tools:[/cyan] {mcp_tools_count}")
    print(f"[cyan]Total tools:[/cyan] {local_count + mcp_tools_count}")
    # List some tool names
    local_tools, mcp_tools = agent.get_tool_names()
    if local_tools:
        print(f"\n[green]Local tools:[/green] {local_tools}")
    if mcp_tools:
        print(f"[green]MCP tools:[/green] {mcp_tools}")
    print()


async def async_main() -> None:
    """Main REPL function"""
    model_id = "openai.gpt-oss-20b-1:0"
    mcp_config_filepath = find_mcp_config_file()

    print("[bold cyan]Starting z007 - Fast Micro Agent...[/bold cyan]")
    logger.info("[REPL] Starting agent main loop.")
    logger.info(f"Model: {model_id}")
    if mcp_config_filepath:
        logger.info(f"Using MCP config: {mcp_config_filepath}")
    else:
        logger.info("No MCP config file found")

    # Create tools and agent
    tools = create_tools()

    try:
        async with Agent(
            model_id=model_id,
            system_prompt="You are a helpful assistant with access to various tools. Be concise but informative in your responses.",
            tools=tools,
            mcp_config=load_mcp_config_from_file(mcp_config_filepath) if mcp_config_filepath else None,
            max_turns=10
        ) as agent:
            
            # Show initial info
            local_count, mcp_server_count, mcp_tools_count = agent.get_tool_counts()
            print(f"[bold yellow]Tools:[/bold yellow] {local_count} local + {mcp_tools_count} MCP from {mcp_server_count} servers = {local_count + mcp_tools_count} total")
            print_help()
            
            # Main REPL loop
            conversation_history = []
            while True:
                try:
                    logger.info("[REPL] Waiting for user input...")
                    user_input = console.input("[bold green]>>> [/bold green]").strip()
                    logger.info(f"[REPL] User input: {user_input}")
                    
                    if not user_input:
                        continue
                        
                    # Handle commands
                    if user_input.startswith('/'):
                        command = user_input.lower()
                        logger.info(f"[REPL] Command received: {command}")
                        if command in ['/exit']:
                            logger.info("[REPL] Exiting REPL loop.")
                            print("[bold red]Goodbye![/bold red]")
                            break
                        elif command == '/help':
                            print_help()
                            continue
                        elif command == '/tools':
                            print_tools_info(agent)
                            continue
                        elif command == '/clear':
                            conversation_history.clear()
                            print("[yellow]Conversation history cleared.[/yellow]")
                            continue
                        else:
                            logger.warning(f"[REPL] Unknown command: {user_input}")
                            print(f"[red]Unknown command:[/red] {user_input}. Type /help for available commands.")
                            continue
                    
                    # Process user message
                    print("[italic]Thinking...[/italic]")
                    logger.info("[REPL] Running agent conversation...")
                    responses = await agent.run_conversation(user_input)
                    last_response = responses[-1] if responses else None
                    
                    # Extract and display answer
                    answer = Agent.extract_final_answer(last_response) if last_response else 'No response'
                    logger.info(f"[REPL] Assistant response: {answer}")
                    print(f"\n[bold blue]Assistant:[/bold blue] {answer}")
                    
                    # Show tools used (if any)
                    tools_used = get_called_tools(responses)
                    if tools_used:
                        logger.info(f"[REPL] Tools used: {', '.join(tools_used)}")
                        print(f"[magenta]Tools used:[/magenta] {', '.join(tools_used)}")
                    print()  # Add spacing
                    
                    # Store in conversation history for reference
                    conversation_history.append({
                        'user': user_input,
                        'assistant': answer,
                        'tools': tools_used
                    })
                    
                except KeyboardInterrupt:
                    logger.info("[REPL] KeyboardInterrupt received.")
                    print("\n\n[bold red]Use /exit to leave the REPL.[/bold red]")
                    continue
                except EOFError:
                    logger.info("[REPL] EOFError received. Exiting.")
                    print("\n[bold red]Goodbye![/bold red]")
                    break
                except Exception as e:
                    logger.error(f"Error processing input: {e}")
                    print(f"[red]Error:[/red] {e}")
                    continue

    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        print(f"[bold red]Failed to start agent:[/bold red] {e}")
        sys.exit(1)


def main() -> None:
    """Entry point"""
    anyio.run(async_main)


if __name__ == "__main__":
    main()
