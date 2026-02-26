# PageIndex Reasoning RAG Demo

This project demonstrates a **vectorless, reasoning-based** approach to Retrieval Augmented Generation (RAG). Instead of traditional semantic similarity search, it uses an LLM to navigate a hierarchical "Tree Index" of a document.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in this directory with your Google API Key:
```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run the App
```bash
streamlit run app.py
```

## 🧠 How it Works
1. **Indexing**: The document (PDF/DOCX) is parsed and summarized page-by-page. These pages are then grouped into sections, each with its own LLM-generated summary, creating a tree.
2. **Retrieval**: When you ask a question, the LLM reasons over the high-level section summaries to decide where to look, then drills down into specific pages.
3. **Traceability**: You can see the actual "reasoning path" the LLM took to find your answer.
