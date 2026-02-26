import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables FIRST
load_dotenv()

import json
import asyncio
from index_builder import IndexBuilder
from retriever import PageRetriever
import tempfile

st.set_page_config(page_title="PageIndex Demo", page_icon="🌳", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .trace-box { background-color: #1e1e1e; border-radius: 10px; padding: 15px; border-left: 5px solid #00f2fe; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌳 PageIndex Reasoning RAG")
st.write("Upload a document and ask questions. The LLM will navigate the tree structure to find answers.")

# Check for API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "your_gemini_api_key_here":
    st.error("⚠️ **API Key Missing!** Please set your `GOOGLE_API_KEY` in the `.env` file inside the `page_index` folder.")
    st.info("You can get an API key for free at [Google AI Studio](https://aistudio.google.com/).")
    st.stop()

# Sidebar for controls
with st.sidebar:
    st.header("1. Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["google-gla:gemini-2.0-flash", "google-gla:gemini-1.5-flash", "google-gla:gemini-1.5-pro"],
        index=0
    )
    
    st.header("2. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    
    if uploaded_file:
        index_filename = f"{uploaded_file.name}_index.json"
        index_path = index_filename 
        
        if st.button("Build Tree Index") or not os.path.exists(index_path):
            with st.spinner("Building Intelligent Tree Index (LLM-powered)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                builder = IndexBuilder(api_key=api_key, model_name=model_choice)
                # Run async build_index in a sync wrapper for Streamlit
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                root = loop.run_until_complete(builder.build_index(tmp_path))
                builder.save_index(root, index_path)
                os.unlink(tmp_path)
                st.success("Index Ready!")
        
        st.session_state.active_index = index_path
        
        st.header("3. Tree Structure")
        with open(st.session_state.active_index, 'r') as f:
            tree = json.load(f)
            st.json(tree)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "trace" in message:
            with st.expander("🔍 Retrieval Trace"):
                for step in message["trace"]:
                    st.write(f"• {step}")

if prompt := st.chat_input("Ask a question about your document..."):
    if 'active_index' not in st.session_state:
        st.error("Please upload and index a document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever = PageRetriever(st.session_state.active_index, api_key=api_key, model_name=model_choice)
            # Run async retrieve in sync wrapper
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(retriever.retrieve(prompt))
            
            if results:
                full_response = ""
                traces = []
                for res in results:
                    full_response += f"### {res['title']}\\n{res['content']}\\n\\n"
                    traces.extend(res['trace'])
                
                st.markdown(full_response)
                with st.expander("🔍 Retrieval Trace"):
                    for step in traces:
                        st.write(f"• {step}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "trace": traces
                })
            else:
                resp = "I couldn't find relevant sections. The LLM reasoning didn't find a path to the answer."
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
