#!/usr/bin/env python3
"""
Test script to verify conversation history functionality
"""

import asyncio

from z007.agent import Agent, create_calculator_tool


async def test_conversation_history():
    """Test that conversation history is properly maintained outside the agent"""

    # Use the built-in calculator tool from z007
    tools = [create_calculator_tool()]

    # Create agent
    async with Agent(
        model_id="openai.gpt-oss-20b-1:0",
        system_prompt="You are a helpful assistant. Be concise.",
        tools=tools,
        max_turns=3,
    ) as agent:
        # Test conversation continuity
        print("=== Testing Conversation History Continuity ===")

        # First interaction
        print("\n1. First interaction:")
        conversation_history = []
        responses1, conversation_history = await agent.run_conversation("Hello, what's 5 + 3?", conversation_history)
        answer1 = Agent.extract_final_answer(responses1[-1]) if responses1 else "No response"
        print(f"User: Hello, what's 5 + 3?")
        print(f"Assistant: {answer1}")
        print(f"History length: {len(conversation_history)}")

        # Second interaction - should remember the context
        print("\n2. Second interaction (should remember context):")
        responses2, conversation_history = await agent.run_conversation(
            "What was my previous question?", conversation_history
        )
        answer2 = Agent.extract_final_answer(responses2[-1]) if responses2 else "No response"
        print(f"User: What was my previous question?")
        print(f"Assistant: {answer2}")
        print(f"History length: {len(conversation_history)}")

        # Third interaction - with fresh agent but same history
        print("\n3. Third interaction (new agent, same history):")
        async with Agent(
            model_id="openai.gpt-oss-20b-1:0",
            system_prompt="You are a helpful assistant. Be concise.",
            tools=tools,
            max_turns=3,
        ) as new_agent:
            responses3, conversation_history = await new_agent.run_conversation(
                "And what was my calculation result?", conversation_history
            )
            answer3 = Agent.extract_final_answer(responses3[-1]) if responses3 else "No response"
            print(f"User: And what was my calculation result?")
            print(f"Assistant: {answer3}")
            print(f"History length: {len(conversation_history)}")

        # Test clearing history
        print("\n4. After clearing history:")
        conversation_history.clear()
        responses4, conversation_history = await agent.run_conversation(
            "What was my previous question?", conversation_history
        )
        answer4 = Agent.extract_final_answer(responses4[-1]) if responses4 else "No response"
        print(f"User: What was my previous question?")
        print(f"Assistant: {answer4}")
        print(f"History length: {len(conversation_history)}")

        print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_conversation_history())
