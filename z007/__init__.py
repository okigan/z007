"""
z007

A lightning-fast tool for interacting with multiple LLM providers with tool support and MCP integration.
"""

from .agent import Agent, create_calculator_tool

__version__ = "0.1.10"
__all__ = ["Agent", "create_calculator_tool"]
