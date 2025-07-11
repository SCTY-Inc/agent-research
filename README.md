# Deep Research Agent with Agno 1.7.0

A professional-grade implementation of OpenAI's Deep Research API cookbook examples using Agno 1.7.0 framework. Features the same sophisticated prompting, structured outputs, citation support, and elegant code architecture.

## Features

- **Multi-Agent Pipeline**: Triage → Clarification → Instruction → Deep Research
- **Professional Prompts**: OpenAI cookbook-grade prompting for superior research quality
- **Structured Outputs**: Type-safe Pydantic models eliminate regex parsing
- **Citation Support**: Extracts and displays sources from research results
- **Tiered Models**: Uses `o3-deep-research-2025-06-26` for research and `gpt-4o-mini` for coordination
- **Interactive UI**: Rich console interface with animated spinners
- **Organized Output**: Saves research reports to `/reports/` directory with timestamps

## Quick Start

### Installation

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with UV
uv sync

# Set up environment
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Usage

```bash
# Run the optimized version
python main.py
```

## Implementation Highlights

### Current Architecture (165 lines)

```python
# Professional prompts from OpenAI cookbook
CLARIFYING_PROMPT = """
Guidelines for concise, friendly clarification questions...
"""

RESEARCH_PROMPT = """
Guidelines for detailed research instructions with tables, 
formatting, language handling, and source prioritization...
"""

# Structured output models
class TriageResponse(BaseModel): needs_clarification: bool
class ClarifyResponse(BaseModel): questions: List[str]
class InstructionResponse(BaseModel): instructions: str

# Type-safe agent configuration with professional prompts
'triage': Agent(model=BASE_MODEL, response_model=TriageResponse, structured_outputs=True)
'clarify': Agent(model=BASE_MODEL, instructions=CLARIFYING_PROMPT, response_model=ClarifyResponse, structured_outputs=True)
'instruction': Agent(model=BASE_MODEL, instructions=INSTRUCTION_PROMPT, response_model=InstructionResponse, structured_outputs=True)
'research': Agent(model=RESEARCH_MODEL, instructions=RESEARCH_PROMPT, tools=[{"type": "web_search_preview"}], markdown=True)

# Elegant pipeline with function separation
async def handle_clarification(query):
    # Dedicated clarification logic
    
async def research_pipeline(query):
    if triage_result.content.needs_clarification:
        query = await handle_clarification(query)
    instructions = await agents['instruction'].run(query)  # Generate detailed instructions
    result = await agents['research'].run(instructions)  # Research with instructions
```

### Key Optimizations

1. **Professional Prompts**: OpenAI cookbook-grade instructions for superior research quality
2. **Structured Outputs**: Replaced regex parsing with Pydantic models
3. **Function Separation**: Extracted clarification logic for better maintainability  
4. **Type Safety**: Full validation of agent responses
5. **Citation Extraction**: Handles both OpenAI and MCP citation formats
6. **Elegant Control Flow**: Clean conditional logic with early returns
7. **Organized Output**: Auto-creates `/reports/` directory for all research files

### Models Used

- **Triage/Clarification**: `gpt-4o-mini` - Fast, efficient for coordination tasks
- **Deep Research**: `o3-deep-research-2025-06-26` - Specialized research model with web search

## Example Output

```
🔬 Deep Research: AI trends in advertising

🔍 Analyzing query...
📝 Getting clarifications...

📋 Please provide context:

1. What industry focus are you most interested in?
> Technology and creative agencies

✅ Got 1 clarifications

🔬 Conducting deep research... (2-3 minutes)

==================================================
📋 RESULTS
==================================================
[Comprehensive research report with analysis, trends, and insights...]
==================================================

📖 SOURCES (12 found):
- Adobe's AI Creative Suite Updates: https://blog.adobe.com/ai-creative-2025
- Meta's Advertising AI Roadmap: https://about.fb.com/news/2025/ad-ai-features
- WPP AI Investment Strategy: https://www.wpp.com/news/ai-transformation-2025
...

💾 Save to file? (Y/n): Y
✅ Saved to reports/report_20250627_1435.md
```

## Technical Evolution

### Before vs After

**Original OpenAI Cookbook**: 139 lines with manual string parsing  
**Reference**: https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents

**Current Implementation**: 165 lines with complete 4-agent pipeline and structured outputs

**Key Improvements**:
- ✅ **Professional-grade prompting** from OpenAI cookbook for superior research quality
- ✅ **Type-safe responses** with Pydantic validation
- ✅ **No regex parsing** - eliminated fragile string operations
- ✅ **Better separation of concerns** with dedicated functions
- ✅ **Enhanced reliability** through structured outputs
- ✅ **Citation support** with multiple format handling

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional - for enhanced logging
AGNO_LOG_LEVEL=INFO
```

### Customization

```python
# Modify models in main.py
BASE_MODEL = OpenAIChat(id="gpt-4o-mini")
RESEARCH_MODEL = OpenAIResponses(id="o3-deep-research-2025-06-26")

# Add custom structured output models
class CustomResponse(BaseModel):
    field: str
```

## Development

### Testing

```bash
# Test imports and structure
python -c "import main; print('✅ All systems ready')"

# Validate structured outputs
python -c "from main import TriageResponse; print('✅ Pydantic models working')"
```

### Dependencies

- **agno>=1.7.0**: Core agent framework with structured output support
- **openai>=1.92.2**: OpenAI API with deep research models
- **pydantic>=2.11.7**: Data validation and structured outputs
- **python-dotenv>=1.1.1**: Environment variable management
- **rich>=14.0.0**: Enhanced console UI and formatting

## Contributing

1. Maintain structured output patterns
2. Preserve the 4-agent pipeline architecture (Triage → Clarify → Instruction → Research)
3. Keep citation functionality intact
4. Test with diverse research topics
5. Follow type-safe practices with Pydantic

## License

MIT License - See original OpenAI cookbook for reference implementation.

## Acknowledgments

- **OpenAI Deep Research API cookbook examples**: https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents
- **Agno 1.7.0 framework** for structured agent orchestration
- **Rich library** for enhanced console UI
- **Pydantic** for robust data validation

## References

- [OpenAI Deep Research API with Agents](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents) - Original cookbook implementation
- [OpenAI Deep Research API Introduction](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api) - API overview