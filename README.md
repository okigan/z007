# ⚡ z007 🤖: Nimble AI Agent
_pronounced: "zee-double-oh-seven"_ 

A lightweight and readable agent for interacting with LLM on AWS Bedrock with tool and MCP (Model Context Protocol) support.

## Features

- 🟢 **Ultra Readable**: Clean, maintainable codebase in about 600 lines - easy to understand, modify, and extend
- ⚡ **Super easy**: Just run `uvx z007@latest`  with `AWS_PROFILE=<your profile>` in env and start chatting instantly  
- ⚡ **Simple Install**: Quick install  `uv tool install --upgrade z007` and start chatting instantly `z007` with `AWS_PROFILE=<your profile>` in env
- 🔧 **Tool Support**: Built-in calculator and easily use plain python functions as tools
- 🔌 **MCP Integration**: Connect to Model Context Protocol servers
- 🐍 **Python API**: Easy integration into your Python projects
- 🚀 **Async**: Concurrent tool execution

## Quick Start

### Install and run with uvx (recommended)

```bash
```bash
# Install and run directly with AWS_PROFILE configured - fastest way to start!
AWS_PROFILE=your-profile uvx z007@latest

# Or install globally
uv tool install z007
AWS_PROFILE=your-profile z007
```


![demo gif](./doc/demo.gif "Optional title text")


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
AWS_PROFILE=your-profile z007 --model-id "openai.gpt-oss-120b-1:0"

# With MCP configuration
z007 --mcp-config ./mcp.json
```

### Python API

#### Simple usage

```python
import asyncio
from z007 import Agent, create_calculator_tool

async def main():
    calculator = create_calculator_tool()
    async with Agent(model_id="openai.gpt-oss-20b-1:0", tools=[calculator]) as agent:
        response = await agent.run("What is 2+2?")
    print(response)

asyncio.run(main())
```

#### Using the Agent class

```python
import asyncio
from z007 import Agent, create_calculator_tool

async def main():
    calculator = create_calculator_tool()
    async with Agent(
        model_id="openai.gpt-oss-20b-1:0",
        system_prompt="You are a helpful coding assistant.",
        tools=[calculator]
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

1. Create `mcp.json`:

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
    },
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    }
  }
}
```

2. Use with z007:

```bash
z007 --mcp-config mcp.json
```

Or in Python:

```python
import json
from z007 import Agent

# Load MCP config
with open("mcp.json") as f:
    mcp_config = json.load(f)

async with Agent(
    model_id="openai.gpt-oss-20b-1:0",
    mcp_config=mcp_config
) as agent:
    response = await agent.run("Search for recent news about AI")
    print(response)
```

## Configuration

### Environment Variables

For AWS Bedrock (default provider):
- `AWS_PROFILE`: AWS profile name (e.g., `AWS_PROFILE=codemobs`)

  **or**

- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key

### Supported Models

AWS Bedrock models with verified access:
- `openai.gpt-oss-20b-1:0` (default)

Note: Model availability depends on your AWS account's Bedrock access permissions. Use `AWS_PROFILE=your-profile` to specify credentials.
- Any AWS Bedrock model with tool support

## Interactive Commands

When running `z007` in interactive mode:

- `/help` - Show help
- `/tools` - List available tools  
- `/clear` - Clear conversation history
- `/exit` - Exit

## Requirements

- Python 3.9+
- LLM provider credentials (AWS for Bedrock)

## License

MIT License
