"""
zappy - Fast Micro Agent

A lightning-fast tool for interacting with multiple LLM providers with tool support and MCP integration.
"""

from .agent import Agent, ToolRegistry, quick_ask, create_calculator_tool

__version__ = "0.1.0"
__all__ = ["Agent", "ToolRegistry", "quick_ask", "create_calculator_tool"]
