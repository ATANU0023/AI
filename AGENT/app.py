import streamlit as st
import os
import asyncio
import importlib.util
from meta_agent import build_agent, list_agents

# --- Initial Page Config ---
st.set_page_config(page_title="AgentCraft Builder", page_icon="🤖", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "create"
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None

# --- Sidebar: Navigation & Agent List ---
with st.sidebar:
    st.title("🤖 AgentCraft")
    st.markdown("---")
    
    # Navigation Buttons
    if st.button("➕ Create New Agent"):
        st.session_state.page = "create"
        st.session_state.selected_agent = None
        st.rerun()
        
    st.markdown("---")
    st.subheader("Your Agents")
    
    available_agents = list_agents()
    
    if not available_agents:
        st.info("No agents built yet.")
    else:
        for agent in available_agents:
            if st.button(f"📄 {agent['name']}", key=agent['file']):
                st.session_state.selected_agent = agent
                st.session_state.page = "test"
                st.rerun()

    st.markdown("---")
    if st.button("🔄 Refresh List"):
        st.rerun()

# --- Utility Functions ---
def run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def get_agent_instance(agent_file):
    import sys
    module_name = "custom_agent"
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, agent_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.agent
    except Exception as e:
        st.error(f"Error loading agent: {e}")
        return None

# --- View: Create Agent ---
def view_create():
    st.header("⚒️ Build a New Agent")
    st.write("Enter a description below and the Meta-Agent will design and generate a custom agent for you.")
    
    user_request = st.text_area("What kind of agent do you need?", 
                               placeholder="e.g., A research assistant that finds tech news and summarizes them.",
                               height=200)
    
    if st.button("Generate Agent"):
        if user_request:
            with st.spinner("Meta-Agent is designing your builder..."):
                try:
                    filepath = run_async(build_agent(user_request))
                    st.success(f"Agent generated successfully!")
                    st.balloons()
                    # Refresh to show in sidebar
                    st.rerun()
                except Exception as e:
                    st.error(f"Build failed: {e}")
        else:
            st.warning("Please enter a description.")

# --- View: Test Agent ---
def view_test():
    agent_info = st.session_state.selected_agent
    
    # Back button and Header
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("🔙 Back"):
            st.session_state.page = "create"
            st.session_state.selected_agent = None
            st.rerun()
    
    with col_title:
        st.header(f"💬 Testing: {agent_info['name']}")

    # Tabs for Interaction and Code View
    tab1, tab2 = st.tabs(["Playground", "Source Code"])
    
    with tab1:
        # Chat Interface
        if "messages" not in st.session_state or st.session_state.get('last_agent') != agent_info['name']:
            st.session_state.messages = []
            st.session_state.last_agent = agent_info['name']

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"Interacting with {agent_info['name']}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    agent_instance = get_agent_instance(agent_info['file'])
                    if agent_instance:
                        try:
                            result = run_async(agent_instance.run(prompt))
                            response = result.output
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Agent error: {e}")
    
    with tab2:
        st.subheader("Agent Code")
        with open(agent_info['file'], 'r') as f:
            st.code(f.read(), language='python')

# --- Main Logic Router ---
if st.session_state.page == "create":
    view_create()
elif st.session_state.page == "test" and st.session_state.selected_agent:
    view_test()
else:
    st.session_state.page = "create"
    st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using PydanticAI and Streamlit by Atanu")
