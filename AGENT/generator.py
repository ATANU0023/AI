import os
from jinja2 import Template

AGENT_TEMPLATE = """
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
    os.getenv("GENERATED_AGENT_MODEL", '{{ model_name }}'),
    system_prompt=\"\"\"
    {{ system_prompt }}
    \"\"\",
    deps_type=dict,
)

{% for tool in tools %}
@agent.tool
async def {{ tool }}(ctx: RunContext[dict]) -> str:
    \"\"\"{{ tool_descriptions[tool] }}\"\"\"
    return str(TOOL_MAP['{{ tool }}']())
{% endfor %}

if __name__ == "__main__":
    import asyncio
    async def main():
        prompt = input("Enter prompt for {{ agent_name }}: ")
        result = await agent.run(prompt)
        print(f"\\n--- {{ agent_name }} Response ---\\n")
        print(result.output)
    
    asyncio.run(main())
"""

def generate_agent_code(agent_name, system_prompt, tools, tool_descriptions, model_name="google-gla:gemini-2.5-flash"):
    template = Template(AGENT_TEMPLATE)
    code = template.render(
        agent_name=agent_name,
        system_prompt=system_prompt,
        tools=tools,
        tool_descriptions=tool_descriptions,
        model_name=model_name
    )
    return code

def save_agent(agent_name, code):
    filename = f"{agent_name.lower().replace(' ', '_')}_agent.py"
    # Save in the 'agents' subfolder
    agents_dir = os.path.join(os.getcwd(), "agents")
    if not os.path.exists(agents_dir):
        os.makedirs(agents_dir)
    
    filepath = os.path.join(agents_dir, filename)
    with open(filepath, 'w') as f:
        f.write(code)
    return filepath
