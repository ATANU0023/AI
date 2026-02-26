
import os
import sys
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# Add parent directory to path so we can find tools.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import TOOL_MAP

# Load environment variables
load_dotenv()

# Define the Agent
agent = Agent(
    os.getenv("GENERATED_AGENT_MODEL", 'google-gla:gemini-2.5-flash'),
    system_prompt="""
    You are a highly skilled Python programmer and assistant. Your goal is to write clean, efficient, and well-commented Python code based on user requests. You should always strive to understand the full requirements before writing code, and ask clarifying questions if necessary. You can also help with debugging and refactoring existing Python code.
    """,
    deps_type=dict,
)


@agent.tool
async def search_web(ctx: RunContext[dict]) -> str:
    """Use this tool to search for Python documentation, examples, common patterns, or solutions to programming problems. This is useful for understanding specific library functions, syntax, or algorithms."""
    return str(TOOL_MAP['search_web']())

@agent.tool
async def read_file(ctx: RunContext[dict]) -> str:
    """Use this tool to read the content of a file. This is useful when the user provides existing code for review, debugging, or modification, or when requirements are detailed in a text file."""
    return str(TOOL_MAP['read_file']())

@agent.tool
async def write_file(ctx: RunContext[dict]) -> str:
    """Use this tool to write the generated Python code or any other text output to a specified file. This is crucial for delivering the code to the user in a persistent format."""
    return str(TOOL_MAP['write_file']())


if __name__ == "__main__":
    import asyncio
    async def main():
        prompt = input("Enter prompt for Python Code Writer: ")
        result = await agent.run(prompt)
        print(f"\n--- Python Code Writer Response ---\n")
        print(result.output)
    
    asyncio.run(main())