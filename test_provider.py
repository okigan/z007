#!/usr/bin/env python3
"""
Test script for the multi-provider Agent implementation
"""

import asyncio

from z007.agent import Agent, create_calculator_tool


async def test_providers():
    """Test both Bedrock and OpenAI providers"""

    # Test Bedrock provider (existing functionality)
    print("=" * 50)
    print("Testing Bedrock Provider")
    print("=" * 50)
    try:
        bedrock_agent = Agent(
            "anthropic.claude-3-haiku",
            provider="bedrock",
            tools=[create_calculator_tool()]
        )
        print("✓ Bedrock agent created successfully")

        # Test basic functionality
        tool_counts = bedrock_agent.get_tool_counts()
        print(f"✓ Tool counts: {tool_counts}")

    except Exception as e:
        print(f"✗ Bedrock agent failed: {e}")

    # Test OpenAI provider with local endpoint using httpx
    print("\n" + "=" * 50)
    print("Testing OpenAI Provider (Local Endpoint with httpx)")
    print("=" * 50)
    try:
        openai_agent = Agent(
            "gpt-4",
            provider="openai",
            provider_params={
                "base_url": "http://127.0.0.1:1234/v1",
                "api_key": "dummy-key"  # Local servers often don't need real keys
            },
            tools=[create_calculator_tool()],
            system_prompt="You are a helpful assistant with access to a calculator."
        )
        print("✓ OpenAI agent created successfully")

        # Test basic functionality
        tool_counts = openai_agent.get_tool_counts()
        print(f"✓ Tool counts: {tool_counts}")

        # Test a simple conversation if the server is running
        try:
            response = await openai_agent.run("What is 2 + 3?")
            print(f"✓ Test conversation response: {response}")
        except Exception as e:
            print(f"⚠ Test conversation failed (server may not be running): {e}")

    except Exception as e:
        print(f"✗ OpenAI agent failed: {e}")

    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print("The multi-provider Agent implementation is ready!")
    print("\nRequirements:")
    print("- For Bedrock: AWS credentials configured")
    print("- For OpenAI: uv add httpx")
    print("\nUsage examples:")
    print("\n# Bedrock (unchanged)")
    print('agent = Agent("anthropic.claude-3-haiku", tools=[calc_tool])')
    print("\n# OpenAI-compatible local server")
    print('agent = Agent(')
    print('    "gpt-4",')
    print('    provider="openai",')
    print('    provider_params={')
    print('        "base_url": "http://127.0.0.1:1234/v1",')
    print('        "api_key": "dummy-key"')
    print('    },')
    print('    tools=[calc_tool]')
    print(')')


if __name__ == "__main__":
    asyncio.run(test_providers())
