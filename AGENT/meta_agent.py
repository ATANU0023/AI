import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from generator import generate_agent_code, save_agent

# Load environment variables
load_dotenv()

def get_model():
    provider = os.getenv("AGENT_PROVIDER", "gemini").lower()
    model_name = os.getenv("META_AGENT_MODEL", "gemini-2.5-flash")
    
    if provider == "gemini":
        # pydanticai uses 'google-gla' for Generative Language API (Gemini)
        return f"google-gla:{model_name}"
    elif provider == "openai":
        return f"openai:{model_name}"
    elif provider == "anthropic":
        return f"anthropic:{model_name}"
    return model_name

def get_generated_model():
    provider = os.getenv("AGENT_PROVIDER", "gemini").lower()
    model_name = os.getenv("GENERATED_AGENT_MODEL", "gemini-2.5-flash")
    
    if provider == "gemini":
        return f"google-gla:{model_name}"
    elif provider == "openai":
        return f"openai:{model_name}"
    elif provider == "anthropic":
        return f"anthropic:{model_name}"
    return model_name

def list_agents():
    """Returns a list of all generated agents in the agents/ directory."""
    agents = []
    import os
    agents_dir = os.path.join(os.getcwd(), "agents")
    if not os.path.exists(agents_dir):
        return []
        
    for file in os.listdir(agents_dir):
        if file.endswith("_agent.py") and file != "meta_agent.py":
            # Convert filename like 'joke_weaver_agent.py' to 'Joke Weaver'
            name = file.replace("_agent.py", "").replace("_", " ").title()
            # Store the relative path from the AGENT root
            agents.append({"name": name, "file": os.path.join("agents", file)})
    return agents

# Meta-Agent Schema: This is what the Meta-Agent produces
class AgentBlueprint(BaseModel):
    agent_name: str = Field(description="The name of the agent to build")
    system_prompt: str = Field(description="The system prompt defining the agent's behavior")
    tools: list[str] = Field(description="List of tool names to include. Options: get_current_time, search_web, read_file, write_file")
    tool_descriptions: dict[str, str] = Field(description="Descriptions for each tool in the agent's context")

# Define the Meta-Agent
meta_agent = Agent(
    get_model(),
    output_type=AgentBlueprint,
    system_prompt="""
    You are a Meta-Agent Builder. Your job is to design specialized AI agents based on user requirements.
    You must provide:
    1. A short, descriptive name for the agent.
    2. A detailed system prompt that defines its persona and goals.
    3. A selection of tools from the following list: get_current_time, search_web, read_file, write_file.
    4. Clear descriptions for how the agent should use those tools.
    """,
)

async def build_agent(user_request: str):
    model_id = get_model()
    print(f"\\n--- Analysis: Designing Agent for '{user_request}' using model '{model_id}' ---")
    
    # Run the Meta-Agent to get the blueprint
    try:
        result = await meta_agent.run(user_request)
        blueprint = result.output
    except ModelHTTPError as e:
        print(f"\\n--- Model Error ---")
        print(f"Status Code: {e.status_code}")
        print(f"Model Name: {e.model_name}")
        print(f"Response Body: {e.body}")
        return
    except Exception as e:
        print(f"\\n--- Error ---")
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        return
    
    print(f"\\nBlueprint Created:")
    print(f"- Name: {blueprint.agent_name}")
    print(f"- Tools: {', '.join(blueprint.tools)}")
    
    # Generate the code
    code = generate_agent_code(
        agent_name=blueprint.agent_name,
        system_prompt=blueprint.system_prompt,
        tools=blueprint.tools,
        tool_descriptions=blueprint.tool_descriptions,
        model_name=get_generated_model()
    )
    
    # Save to file
    filepath = save_agent(blueprint.agent_name, code)
    print(f"\\n--- Success! Agent generated at: {filepath} ---")
    return filepath

if __name__ == "__main__":
    import asyncio
    async def main():
        print("Welcome to AgentCraft - The Meta-Agent Builder")
        # For non-interactive test
        request = "create a joke maker agent" 
        await build_agent(request)
    
    asyncio.run(main())
