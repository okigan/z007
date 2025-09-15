"""
z007

A lightning-fast tool for interacting with multiple LLM providers with tool support and MCP integration.
"""

from .agent import Agent, create_calculator_tool, get_called_tools

__version__ = "0.2.1"
__all__ = ["Agent", "create_calculator_tool", "get_called_tools"]
