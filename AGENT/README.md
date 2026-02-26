# AgentCraft: The Meta-Agent Builder

AgentCraft is a system that allows you to generate specialized AI agents based on natural language descriptions. You describe what you need, and the Meta-Agent designs and generates a functional Python script for you.

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard
Run the Streamlit interface:
```bash
python -m streamlit run app.py
```

### 3. Run Your New Agent
The script will generate a new file (e.g., `researcher_agent.py`). Run it directly:
```bash
python researcher_agent.py
```

## 🛠️ Components

- **`meta_agent.py`**: The orchestrator. It uses Gemini to turn your request into an agent blueprint and triggers the code generator.
- **`generator.py`**: A template-based engine (using Jinja2) that writes the Python code for your new agent.
- **`tools.py`**: A shared registry of capabilities your agents can use. Currently includes:
  - `get_current_time`: Get the system time.
  - `search_web`: Mocked web search utility.
  - `read_file` / `write_file`: Local file I/O.

## 🧪 Testing
You can verify the generation logic by running the test script:
```bash
python test_generation.py
```

## 💡 Customization
To add new capabilities to all future agents, simply add new functions to `AGENT/tools.py` and register them in the `TOOL_MAP` dictionary.
