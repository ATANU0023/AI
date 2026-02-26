import os
import httpx
from datetime import datetime

def get_current_time():
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_web(query: str):
    """
    Simulates a web search. In a real scenario, this would call a Search API.
    For this agent builder, we'll provide a placeholder or use a simple API if available.
    """
    return f"Search result for: {query}\n(Mocked result: Found multiple sources regarding {query})"

def read_file(path: str):
    """Reads the content of a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str):
    """Writes content to a file."""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return "File written successfully."
    except Exception as e:
        return f"Error writing file: {e}"

# Mapping of tool names to their functions
TOOL_MAP = {
    "get_current_time": get_current_time,
    "search_web": search_web,
    "read_file": read_file,
    "write_file": write_file
}
