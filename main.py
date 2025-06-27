import os, asyncio
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat, OpenAIResponses
from rich.console import Console

load_dotenv()
console = Console()

# Structured output models
class TriageResponse(BaseModel): needs_clarification: bool
class ClarifyResponse(BaseModel): questions: List[str]

# Models and agents
BASE_MODEL = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
RESEARCH_MODEL = OpenAIResponses(id="o3-deep-research-2025-06-26", api_key=os.getenv("OPENAI_API_KEY"))

agents = {
    'triage': Agent(model=BASE_MODEL, instructions="Analyze if the research query needs clarification.", response_model=TriageResponse, structured_outputs=True),
    'clarify': Agent(model=BASE_MODEL, instructions="Generate 2-3 specific clarifying questions.", response_model=ClarifyResponse, structured_outputs=True),
    'research': Agent(model=RESEARCH_MODEL, instructions="Perform deep empirical research based on the user's instructions.", tools=[{"type": "web_search_preview"}], markdown=True)
}


def show_citations(result):
    """Display citations from research result"""
    citations = []
    if hasattr(result, 'citations') and result.citations:
        citations = [(c.title, c.url) for c in result.citations if hasattr(c, 'url')]
    elif hasattr(result, 'content') and hasattr(result.content, 'annotations'):
        citations = [(getattr(a, 'title', 'Source'), getattr(a, 'url', '')) 
                    for a in (result.content.annotations or []) if getattr(a, 'type', None) == 'url_citation']
    
    print(f"\n📖 SOURCES ({len(citations)} found):")
    for title, url in citations: 
        print(f"- {title}: {url}")
    if not citations: 
        print("- Research completed with web search capabilities")

async def handle_clarification(query):
    """Handle clarification questions and return enhanced query"""
    print("📝 Getting clarifications...")
    clarify_result = await asyncio.to_thread(agents['clarify'].run, query)
    questions = clarify_result.content.questions
    
    if not questions:
        return query
    
    print("\n📋 Please provide context:")
    answers = []
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. {q}")
        if ans := input("> ").strip():
            answers.append(f"{q}: {ans}")
            print(f"   ✓ {ans[:50]}...")
    
    print(f"✅ Got {len(answers)} clarifications\n")
    return f"{query}\n\nContext:\n" + "\n".join(answers) if answers else query

async def research_pipeline(query):
    """Optimized research pipeline with structured outputs"""
    print(f"🔬 Deep Research: {query}\n")
    
    # Triage and conditional clarification
    print("🔍 Analyzing query...")
    triage_result = await asyncio.to_thread(agents['triage'].run, query)
    
    if triage_result.content.needs_clarification:
        query = await handle_clarification(query)
    
    # Research
    with console.status("🔬 Conducting deep research... (2-3 minutes)", spinner="dots"):
        result = await asyncio.to_thread(agents['research'].run, query)
    
    return result

async def main():
    """Main entry point with citations"""
    print("🔬 Agno Deep Research\nUsing o3-deep-research-2025-06-26\n")
    
    if not (query := input("🤔 Research topic? ").strip()):
        return
    
    try:
        result = await research_pipeline(query)
        
        print("\n" + "="*50)
        print("📋 RESULTS")
        print("="*50)
        print(result.content)
        print("="*50)
        
        show_citations(result)
        
        if input("\n💾 Save to file? (Y/n): ").strip().lower() != 'n':
            import os
            os.makedirs("reports", exist_ok=True)
            filename = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            with open(filename, 'w') as f:
                f.write(f"# Research Results\n\n**Query:** {query}\n\n{result.content}")
            print(f"✅ Saved to {filename}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())