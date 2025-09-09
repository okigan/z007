# z007 ⚡ Fast Micro Agent

A lightning-fast, lightweight agent for interacting with LLM providers with built-in tool support and MCP (Model Context Protocol) integration.

## Features

- ⚡ **Lightning Fast**: Quick setup with `uvx z007` - start chatting instantly  
- 🔧 **Tool Support**: Built-in calculator and custom tool integration
- 🔌 **MCP Integration**: Connect to Model Context Protocol servers
- 🎯 **Multiple Providers**: Currently supports AWS Bedrock (more coming soon)
- 🐍 **Python API**: Easy integration into your Python projects
- 🚀 **Async**: Fast, concurrent tool execution

## Quick Start

### Install and run with uvx (recommended)

```bash
# Install and run directly - fastest way to start!
uvx z007

# Or install globally  
uvx install z007
z007
```

### Install as Python package

```bash
pip install z007
```

## Usage

### Command Line

```bash
# Start interactive chat
z007

# With custom model (AWS Bedrock)
z007 --model-id "anthropic.claude-3-sonnet-20240229-v1:0"

# With MCP configuration
z007 --mcp-config .vscode/mcp.json
```

### Python API

#### Simple usage

```python
import asyncio
from z007 import Agent

async def main():
    async with Agent(model_id="openai.gpt-oss-20b-1:0") as agent:
        response = await agent.run("What is 2+2?")
    print(response)

asyncio.run(main())
```

#### With tools

```python
import asyncio
from z007 import Agent, create_calculator_tool

async def main():
    calculator = create_calculator_tool()
    async with Agent(
        model_id="openai.gpt-oss-20b-1:0",
        tools=[calculator]
    ) as agent:
        response = await agent.run("Calculate 15 * 23 + 7")
    print(response)

asyncio.run(main())
```

#### Using the Agent class

```python
import asyncio
from z007 import Agent

async def main():
    async with Agent(
        model_id="openai.gpt-oss-20b-1:0",
        system_prompt="You are a helpful coding assistant."
    ) as agent:
        response = await agent.run("Write a Python function to reverse a string")
        print(response)

asyncio.run(main())
```

### Custom Tools

Create your own tools by writing simple Python functions:

```python
import asyncio
from z007 import Agent

def weather_tool(city: str) -> str:
    """Get weather information for a city"""
    # In a real implementation, call a weather API
    return f"The weather in {city} is sunny, 25°C"

def file_reader_tool(filename: str) -> str:
    """Read contents of a file"""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

async def main():
    async with Agent(
        model_id="openai.gpt-oss-20b-1:0",
        tools=[weather_tool, file_reader_tool]
    ) as agent:
        response = await agent.run("What's the weather like in Paris?")
    print(response)

asyncio.run(main())
```

### MCP Integration

Connect to Model Context Protocol servers for advanced capabilities:

1. Create `.vscode/mcp.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/project"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${env:BRAVE_API_KEY}"
      }
    }
  }
}
```

2. Use with z007:

```bash
z007 --mcp-config .vscode/mcp.json
```

Or in Python:

```python
from z007 import Agent

async with Agent(
    model_id="openai.gpt-oss-20b-1:0",
    mcp_config_path=".vscode/mcp.json"
) as agent:
    response = await agent.run("Search for recent news about AI")
    print(response)
```

## Configuration

### Environment Variables

For AWS Bedrock (default provider):
- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key

### Supported Models

Current AWS Bedrock models:
- `openai.gpt-oss-20b-1:0` (default)
- `meta.llama3-70b-instruct-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`
- Any AWS Bedrock model with tool support

## Interactive Commands

When running `z007` in interactive mode:

- `/help` - Show help
- `/tools` - List available tools  
- `/clear` - Clear conversation history
- `/exit` - Exit

## Development

### Setup
```bash
# Clone and setup
git clone https://github.com/okigan/z007.git
cd z007
uv sync
```

### Running locally
```bash
# Run the CLI
uv run z007

# Run with specific model
uv run z007 --model-id "anthropic.claude-3-sonnet-20240229-v1:0"

# Run examples
uv run python examples.py

# Run tests
uv run python test.py
```

### Building and Publishing
```bash
# Build the package
uv build

# Install locally for testing
uv pip install dist/z007-0.1.0-py3-none-any.whl

# Publish to Test PyPI (recommended first)
uv publish --repository testpypi

# Publish to PyPI
uv publish
```

### Development Tools
```bash
# Type checking
uv run mypy z007/

# Code formatting and linting
uv run ruff check z007/
uv run ruff format z007/

# Install in editable mode for development
uv pip install -e .
```

## API Reference

### `z007.Agent`

Main agent class for LLM interactions.

```python
async with Agent(
    model_id="openai.gpt-oss-20b-1:0",
    system_prompt=None,
    tools=[],
    mcp_config=None,
    max_turns=10
) as agent:
    response = await agent.run("Your question")
```

### `z007.create_calculator_tool()`

Creates a basic calculator tool function.

```python
calculator = create_calculator_tool()
# Use with Agent
```

## Requirements

- Python 3.9+
- LLM provider credentials (AWS for Bedrock)

## License

MIT License

---

**z007** ⚡ Fast, lightweight, powerful. Get things done quickly! 🚀
