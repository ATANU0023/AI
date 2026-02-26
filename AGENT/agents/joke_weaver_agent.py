
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
    You are a comedic genius, a master of wit and wordplay. Your purpose is to craft original, engaging, and genuinely funny jokes on demand. You can create puns, one-liners, short observational humor, or even short comedic scenarios. Always strive for originality and a good laugh.
    """,
    deps_type=dict,
)


@agent.tool
async def search_web(ctx: RunContext[dict]) -> str:
    """This tool allows the Joke Weaver to research current events, popular culture, specific topics requested by the user, or different joke structures and comedic styles to help generate fresh and relevant humor."""
    return str(TOOL_MAP['search_web']())


if __name__ == "__main__":
    import asyncio
    async def main():
        prompt = input("Enter prompt for Joke Weaver: ")
        result = await agent.run(prompt)
        print(f"\n--- Joke Weaver Response ---\n")
        print(result.output)
    
    asyncio.run(main())