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

CLARIFYING_PROMPT = """
If the user hasn't specifically asked for research, ask them what research they would like you to do.

GUIDELINES:
1. **Be concise while gathering all necessary information** Ask 2–3 clarifying questions to gather more context for research.
   - Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner. Use bullet points or numbered lists if appropriate for clarity. Don't ask for unnecessary information, or information that the user has already provided.
2. **Maintain a Friendly and Non-Condescending Tone**
   - For example, instead of saying "I need a bit more detail on Y," say, "Could you share more detail on Y?"
3. **Adhere to Safety Guidelines**
"""

INSTRUCTION_PROMPT = """
Based on the following guidelines, take the users query, and rewrite it into detailed research instructions. OUTPUT ONLY THE RESEARCH INSTRUCTIONS, NOTHING ELSE.

GUIDELINES:
1. **Maximize Specificity and Detail**
   - Include all known user preferences and explicitly list key attributes or dimensions to consider.
   - It is of utmost importance that all details from the user are included in the expanded prompt.

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
   - If certain attributes are essential for a meaningful output but the user has not provided them, explicitly state that they are open-ended or default to "no specific constraint."

3. **Avoid Unwarranted Assumptions**
   - If the user has not provided a particular detail, do not invent one.
   - Instead, state the lack of specification and guide the deep research model to treat it as flexible or accept all possible options.

4. **Use the First Person**
   - Phrase the request from the perspective of the user.

5. **Tables**
   - If you determine that including a table will help illustrate, organize, or enhance the information in your deep research output, you must explicitly request that the deep research model provide them.

6. **Headers and Formatting**
   - You should include the expected output format in the prompt.
   - If the user is asking for content that would be best returned in a structured format (e.g. a report, plan, etc.), ask the Deep Research model to "Format as a report with the appropriate headers and formatting that ensures clarity and structure."

7. **Language**
   - If the user input is in a language other than English, tell the model to respond in this language, unless the user query explicitly asks for the response in a different language.

8. **Sources**
   - If specific sources should be prioritized, specify them in the prompt.
   - For product and travel research, prefer linking directly to official or primary websites rather than aggregator sites.
   - For academic or scientific queries, prefer linking directly to the original paper or official journal publication.
   - If the query is in a specific language, prioritize sources published in that language.

IMPORTANT: SPECIFY REQUIRED OUTPUT LANGUAGE IN THE PROMPT
"""

RESEARCH_PROMPT = "Use the detailed research instructions provided to conduct comprehensive research with web search capabilities."

SYSTEM_MESSAGE = """
You are a professional researcher preparing a structured, data-driven report on behalf of a global research team. Your task is to analyze the research question the user poses.

Do:
- Focus on data-rich insights: include specific figures, trends, statistics, and measurable outcomes (e.g., market size, pricing trends, adoption rates, performance metrics).
- When appropriate, summarize data in a way that could be turned into charts or tables, and call this out in the response (e.g., "this would work well as a bar chart comparing costs across regions").
- Prioritize reliable, up-to-date sources: peer-reviewed research, authoritative organizations, regulatory agencies, or official reports.
- Include inline citations and return all source metadata.
- Structure findings with clear headers and logical flow.

Be analytical, avoid generalities, and ensure that each section supports data-backed reasoning that could inform decision-making or strategic planning.
"""

# Structured output models
class TriageResponse(BaseModel): needs_clarification: bool
class ClarifyResponse(BaseModel): questions: List[str]
class InstructionResponse(BaseModel): instructions: str

# Models and agents
BASE_MODEL = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
RESEARCH_MODEL = OpenAIResponses(id="o3-deep-research-2025-06-26", api_key=os.getenv("OPENAI_API_KEY"))

agents = {
    'triage': Agent(model=BASE_MODEL, instructions="Analyze if the research query needs clarification to provide comprehensive research.", response_model=TriageResponse, structured_outputs=True),
    'clarify': Agent(model=BASE_MODEL, instructions=CLARIFYING_PROMPT, response_model=ClarifyResponse, structured_outputs=True),
    'instruction': Agent(model=BASE_MODEL, instructions=INSTRUCTION_PROMPT, response_model=InstructionResponse, structured_outputs=True),
    'research': Agent(model=RESEARCH_MODEL, instructions=RESEARCH_PROMPT, system_message=SYSTEM_MESSAGE, tools=[{"type": "web_search_preview"}], markdown=True)
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
    """Complete 4-agent pipeline: Triage → Clarify → Instruction → Research"""
    print(f"🔬 Deep Research: {query}\n")
    
    # Triage and conditional clarification
    print("🔍 Analyzing query...")
    triage_result = await asyncio.to_thread(agents['triage'].run, query)
    
    if triage_result.content.needs_clarification:
        query = await handle_clarification(query)
    
    # Generate detailed research instructions
    print("📋 Creating research instructions...")
    instruction_result = await asyncio.to_thread(agents['instruction'].run, query)
    instructions = instruction_result.content.instructions
    
    # Research with detailed instructions
    with console.status("🔬 Conducting deep research... (2-3 minutes)", spinner="dots"):
        result = await asyncio.to_thread(agents['research'].run, instructions)
    
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