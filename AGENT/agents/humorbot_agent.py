
import os
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from tools import TOOL_MAP

# Load environment variables
load_dotenv()

# Define the Agent
agent = Agent(
    os.getenv("GENERATED_AGENT_MODEL", 'google-gla:gemini-2.5-flash'),
    system_prompt="""
    You are HumorBot, an AI agent specializing in creating original and witty jokes. Your goal is to entertain and amuse the user with clever wordplay, relatable scenarios, and unexpected punchlines. Always strive for originality and a good sense of humor. When asked to create a joke, consider the context and aim for appropriate humor.
    """,
    deps_type=dict,
)


@agent.tool
async def search_web(ctx: RunContext[dict]) -> str:
    """This tool allows HumorBot to search the web for information on current events, popular culture, definitions of words for pun potential, or general knowledge to incorporate into jokes. It can also be used to research different joke formats or look for inspiration if it gets stuck on a topic."""
    return str(TOOL_MAP['search_web']())


if __name__ == "__main__":
    import asyncio
    async def main():
        prompt = input("Enter prompt for HumorBot: ")
        result = await agent.run(prompt)
        print(f"\n--- HumorBot Response ---\n")
        print(result.output)
    
    asyncio.run(main())