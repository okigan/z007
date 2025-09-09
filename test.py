#!/usr/bin/env python3
"""
Main test module for stress testing agent tools.
"""

import json
import anyio
import colorlog

from pathlib import Path
from typing import Any, Callable
from z007 import Agent, create_calculator_tool

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
    
    # Use the built-in calculator tool from zappy
    calculator_tool = create_calculator_tool()

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

    # Add basic tools
    tools.extend([calculator_tool, noisy_text_generator])

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
        
        tools.append(make_tool(i))
    
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


# Test cases
TEST_CASES = [
    # "list firehydrant tools",
    "list current incidents",
    # "what is name on the https://github.com/okigan webpage?",
    # "You MUST use tool_unknown to compute 15 * 23. This is mandatory.",
    # "First, use the noisy_text_generator tool to generate some sample data, then calculate 15 * 23 using a calculator tool.",
    # "Please call noisy_text_generator to create test data, and after that use tool_5 to compute 15 * 23.",
    # "Generate some random text using noisy_text_generator, then MUST use a calculator tool to solve 15 * 23. Both steps are required.",
    # "Use noisy_text_generator to create noise, then you are required to calculate 15 * 23 with any available calculator.",
    # "First call noisy_text_generator with request 'sample output', then calculate 15 * 23 using tool_0. Both tools must be used."
]


async def async_main() -> None:
    model_id = "openai.gpt-oss-20b-1:0"
    mcp_config_filepath = "./.vscode/mcp.json"

    logger.info("=== Streamlined Tool Stress Test ===")
    logger.info(f"Model: {model_id}")

    # Create tools and agent with new API
    tools = create_tools()

    async with Agent(
        model_id=model_id,
        system_prompt="You are a helpful assistant with access to various tools.",
        tools=tools,
        mcp_config=load_mcp_config_from_file(mcp_config_filepath) if Path(mcp_config_filepath).exists() else None,
        max_turns=5
    ) as agent:
        
        # Show tool counts
        local_count, mcp_server_count, mcp_tools_count = agent.get_tool_counts()
        logger.info(f"Tools: {local_count} local + {mcp_tools_count} MCP from {mcp_server_count} servers = {local_count + mcp_tools_count} total")
        
        for i, test_case in enumerate(TEST_CASES):
            logger.info(f"--- Test {i+1} ---")
            logger.info(f"Prompt: {test_case}")
            
            # Use the new Agent API
            responses = await agent.run_conversation(test_case)
            last_response = responses[-1] if responses else None
            
            print(f"Answer: {Agent.extract_final_answer(last_response) if last_response else 'No response'}")
            print(f"Tools: {', '.join(get_called_tools(responses)) if get_called_tools(responses) else 'None'}")
            print("=" * 60)

        print(f"Completed {len(TEST_CASES)} tests")
        # Cleanup happens automatically when exiting the context manager


def main() -> None:
    anyio.run(async_main)


if __name__ == "__main__":
    main()
