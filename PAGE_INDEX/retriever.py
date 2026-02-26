import json
from typing import List, Dict, Any
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

class PageRetriever:
    def __init__(self, index_path: str, api_key: str, model_name: str = "google-gla:gemini-2.0-flash"):
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        # We ensure the environment variable is set for the provider
        os.environ["GOOGLE_API_KEY"] = api_key
        self.agent = Agent(model_name)

    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Uses LLM reasoning to navigate the tree.
        """
        results = []
        path_trace = []
        
        # Step 1: Browse Sections
        sections = self.index.get("children", [])
        section_summaries = "\\n".join([f"- {s['title']}: {s['summary']}" for s in sections])
        
        prompt = f"""
        User Query: {query}
        Available Sections: \\n{section_summaries}
        
        Which section(s) are most likely to contain the answer? 
        Return strictly the titles of the sections, separated by a comma. If none, return 'None'.
        """
        
        result = await self.agent.run(prompt)
        selected_titles = [t.strip() for t in result.data.split(",")]
        
        for section in sections:
            if section["title"] in selected_titles:
                path_trace.append(f"Reasoned into: {section['title']}")
                
                # Step 2: Browse Pages within the section
                pages = section.get("children", [])
                page_summaries = "\\n".join([f"- {p['title']}: {p['summary']}" for p in pages])
                
                prompt_p = f"""
                User Query: {query}
                Section Context: {section['title']} - {section['summary']}
                Available Pages: \\n{page_summaries}
                
                Which page(s) contain the specific information?
                Return strictly the titles of the pages, separated by a comma. If none, return 'None'.
                """
                
                result_p = await self.agent.run(prompt_p)
                selected_pages = [t.strip() for t in result_p.data.split(",")]
                
                for page in pages:
                    if page["title"] in selected_pages:
                        results.append({
                            "title": page["title"],
                            "content": page["content"],
                            "trace": list(path_trace) + [f"Extracted from: {page['title']}"]
                        })
        
        return results

if __name__ == "__main__":
    import asyncio
    async def test():
        # retriever = PageRetriever("page_Index/tree_index.json")
        # found = await retriever.retrieve("Tell me about the AI strategy")
        pass
    # asyncio.run(test())
