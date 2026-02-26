import os
from typing import List, Optional, Dict
import json
from pypdf import PdfReader
from docx import Document
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

class PageNode:
    def __init__(self, node_id: str, title: str, summary: str, level: int, content: Optional[str] = None):
        self.node_id = node_id
        self.title = title
        self.summary = summary
        self.level = level # 0: Root, 1: Section, 2: Page/Subsection
        self.content = content
        self.children: List['PageNode'] = []

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "level": self.level,
            "content": self.content if self.content else None,
            "children": [child.to_dict() for child in self.children]
        }

class IndexBuilder:
    def __init__(self, api_key: str, model_name: str = "google-gla:gemini-2.0-flash"):
        # We ensure the environment variable is set for the provider
        os.environ["GOOGLE_API_KEY"] = api_key
        self.agent = Agent(model_name)

    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text())
        return pages

    def extract_text_from_docx(self, file_path: str) -> List[str]:
        doc = Document(file_path)
        # For simplicity, we'll treat each paragraph as a potential chunk or grouping
        # But for the tree, let's group by headers if possible or just chunk
        full_text = [para.text for para in doc.paragraphs if para.text.strip()]
        return full_text

    async def build_index(self, file_path: str) -> PageNode:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            pages = self.extract_text_from_pdf(file_path)
        elif ext == ".docx":
            pages = self.extract_text_from_docx(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                pages = [f.read()]

        root = PageNode("root", os.path.basename(file_path), "Root node for the document", 0)
        
        # Simple clustering/grouping strategy for demo: 
        # 1. Summarize each 'page' or chunk.
        # 2. Group them into 'sections' based on LLM reasoning.
        
        print(f"Summarizing {len(pages)} pages...")
        page_summaries = []
        for i, page_text in enumerate(pages):
            if not page_text.strip(): continue
            prompt = f"Summarize the following document page in one sentence:\\n\\n{page_text[:4000]}"
            result = await self.agent.run(prompt)
            summary = result.data
            node = PageNode(f"p_{i}", f"Page {i+1}", summary, 2, page_text)
            page_summaries.append(node)

        # For the demo, we'll create one 'Section' and put all pages under it, 
        # or we could ask the LLM to group them. Let's keep it simple: group every 5 pages.
        for i in range(0, len(page_summaries), 5):
            chunk = page_summaries[i:i+5]
            section_titles = [n.title for n in chunk]
            section_content = "\\n".join([n.summary for n in chunk])
            prompt = f"Create a section title and one-sentence summary for a group of pages with these summaries:\\n\\n{section_content}"
            result = await self.agent.run(prompt)
            # Assuming LLM returns "Title: Summary"
            res_text = result.data
            title = res_text.split(":")[0] if ":" in res_text else f"Section {i//5 + 1}"
            summary = res_text.split(":")[1] if ":" in res_text else res_text
            
            section_node = PageNode(f"sec_{i//5}", title.strip(), summary.strip(), 1)
            section_node.children.extend(chunk)
            root.children.append(section_node)

        return root

    def save_index(self, root: PageNode, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(root.to_dict(), f, indent=4)

if __name__ == "__main__":
    import asyncio
    async def test():
        builder = IndexBuilder()
        # You would need a real file here for a test
        # root = await builder.build_index("sample.pdf")
        # builder.save_index(root, "page_Index/tree_index.json")
    asyncio.run(test())
