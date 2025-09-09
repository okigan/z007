# z007 ⚡ Fast Micro Agent

*Pronounced as "ze double O 7"*

A lightweight and readable agent for interacting with LLM on AWS Bedrock with tool and MCP (Model Context Protocol) support.

## Features

- 🟢 **Ultra Readable**: Clean, maintainable codebase in ~500 lines - easy to understand, modify, and extend
- ⚡ **Super easy**: Just run `uvx z007`  with `AWS_PROFILE=<your profile>` in env and start chatting instantly  
- ⚡ **Simple Install**: Quick install  `uv tool install --upgrade z007` and start chatting instantly `z007` with `AWS_PROFILE=<your profile>` in env
- 🔧 **Tool Support**: Built-in calculator and easily use plain python functions as tools
- 🔌 **MCP Integration**: Connect to Model Context Protocol servers
- 🐍 **Python API**: Easy integration into your Python projects
- 🚀 **Async**: Concurrent tool execution

## Quick Start

### Install and run with uvx (recommended)

```bash
# Install and run directly - fastest way to start!
uvx z007

# Or install globally  
uv tool install z007
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
z007 --mcp-config ./mcp.json
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
- `AWS_PROFILE`: AWS profile name
  **or**
- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key

### Supported Models

Current AWS Bedrock models:
- `openai.gpt-oss-20b-1:0` (default)
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

---

**z007** ⚡ Fast, lightweight, powerful. Get things done quickly! 🚀
