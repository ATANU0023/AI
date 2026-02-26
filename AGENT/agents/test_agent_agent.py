import os
import sys
from pydantic_ai import Agent, RunContext

# Add parent directory to path so we can find tools.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import TOOL_MAP

# Define the Agent
agent = Agent(
    'google:gemini-1.5-flash',
    system_prompt="""
    You are a helpful test assistant.
    """,
    deps_type=dict,
)


@agent.tool
async def get_current_time(ctx: RunContext[dict]) -> str:
    """Get the current time."""
    return str(TOOL_MAP['get_current_time']())

@agent.tool
async def read_file(ctx: RunContext[dict]) -> str:
    """Read a file."""
    return str(TOOL_MAP['read_file']())


if __name__ == "__main__":
    import asyncio
    async def main():
        prompt = input("Enter prompt for Test Agent: ")
        result = await agent.run(prompt)
        print(f"\n--- Test Agent Response ---\n")
        print(result.data)
    
    asyncio.run(main())