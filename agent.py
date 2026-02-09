"""
FsExplorer Agent using LangChain with DeepSeek.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Load .env
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from models import Action
from fs import (
    read_file,
    grep_file_content,
    glob_paths,
    scan_folder,
    preview_file,
    parse_file,
)

# DeepSeek pricing (per million tokens) - approximation
DEEPSEEK_INPUT_COST_PER_MILLION = 0.14
DEEPSEEK_OUTPUT_COST_PER_MILLION = 0.28

@dataclass
class TokenUsage:
    """Track token usage and costs."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    tool_result_chars: int = 0
    documents_parsed: int = 0
    documents_scanned: int = 0

    def add_api_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.api_calls += 1

    def add_tool_result(self, result: str, tool_name: str) -> None:
        self.tool_result_chars += len(result)
        if tool_name == "parse_file":
            self.documents_parsed += 1
        elif tool_name == "scan_folder":
            self.documents_scanned += result.count("│ [")
        elif tool_name == "preview_file":
            self.documents_parsed += 1

    def _calculate_cost(self) -> tuple[float, float, float]:
        input_cost = (self.prompt_tokens / 1_000_000) * DEEPSEEK_INPUT_COST_PER_MILLION
        output_cost = (self.completion_tokens / 1_000_000) * DEEPSEEK_OUTPUT_COST_PER_MILLION
        return input_cost, output_cost, input_cost + output_cost

    def summary(self) -> str:
        input_cost, output_cost, total_cost = self._calculate_cost()

        return f"""
═══════════════════════════════════════════════════════════════
                      TOKEN USAGE SUMMARY
═══════════════════════════════════════════════════════════════
  API Calls:           {self.api_calls}
  Prompt Tokens:       {self.prompt_tokens:,}
  Completion Tokens:   {self.completion_tokens:,}
  Total Tokens:        {self.total_tokens:,}
───────────────────────────────────────────────────────────────
  Documents Scanned:   {self.documents_scanned}
  Documents Parsed:    {self.documents_parsed}
  Tool Result Chars:   {self.tool_result_chars:,}
───────────────────────────────────────────────────────────────
  Est. Cost (DeepSeek):
    Input:  ${input_cost:.4f}
    Output: ${output_cost:.4f}
    Total:  ${total_cost:.4f}
═══════════════════════════════════════════════════════════════
"""

# Tool registry
TOOLS = {
    "read": read_file,
    "grep": grep_file_content,
    "glob": glob_paths,
    "scan_folder": scan_folder,
    "preview_file": preview_file,
    "parse_file": parse_file,
}

# System prompt
SYSTEM_PROMPT = """
You are FsExplorer, an AI agent that explores filesystems to answer user questions about documents.

## Available Tools

| Tool | Purpose | Parameters |
|------|---------|------------|
| `scan_folder` | **PARALLEL SCAN** - Scan ALL documents in a folder at once | `directory` |
| `preview_file` | Quick preview of a single document (~first page) | `file_path` |
| `parse_file` | **DEEP READ** - Full content of a document | `file_path` |
| `read` | Read a plain text file | `file_path` |
| `grep` | Search for a pattern in a file | `file_path`, `pattern` |
| `glob` | Find files matching a pattern | `directory`, `pattern` |

## Three-Phase Document Exploration Strategy

### PHASE 1: Parallel Scan (Use `scan_folder`)
When you encounter a folder with documents:
1. Use `scan_folder` to scan ALL documents in parallel
2. This gives you a quick preview of every document at once
3. In your **reason**, explicitly list your document categorization:
   - **RELEVANT**: Documents clearly related to the query (list them)
   - **MAYBE**: Documents that might be relevant (list them)
   - **SKIP**: Documents not relevant (list them)

### PHASE 2: Deep Dive (Use `parse_file`)
1. Use `parse_file` on documents marked RELEVANT
2. In your **reason**, explain what key information you found
3. **WATCH FOR CROSS-REFERENCES** - look for mentions like:
   - "See Exhibit A/B/C..."
   - "As stated in the [Document Name]..."
   - "Refer to [filename]..."
   - Document numbers, exhibit labels, or file names
4. In your **reason**, note any cross-references you discovered

### PHASE 3: Backtracking (Revisit if Cross-Referenced)
**CRITICAL**: If a document you're reading references another document that you SKIPPED:
1. In your **reason**, explain: "Found cross-reference to [document] - need to backtrack"
2. Use `preview_file` or `parse_file` to read the referenced document
3. Continue this until all relevant cross-references are resolved

## Providing Detailed Reasoning

Your `reason` field is displayed to the user, so make it informative:
- After scanning: List which documents you're categorizing as RELEVANT/MAYBE/SKIP and why
- After parsing: Summarize key findings and any cross-references discovered
- When backtracking: Explain which reference led you back to a skipped document

## CRITICAL: Citation Requirements for Final Answers

When providing your final answer, you MUST include citations for ALL factual claims:

### Citation Format
Use inline citations in this format: `[Source: filename, Section/Page]`

Example:
> The total purchase price is $125,000,000 [Source: 01_master_agreement.pdf, Section 2.1],
> consisting of $80M cash [Source: 01_master_agreement.pdf, Section 2.1(a)],
> $30M in stock [Source: 10_stock_purchase.pdf, Section 1], and
> $15M in escrow [Source: 09_escrow_agreement.pdf, Section 2].

### Citation Rules
1. **Every factual claim needs a citation** - dates, numbers, names, terms, etc.
2. **Be specific** - include section numbers, article numbers, or page references when available
3. **Use the actual filename** - not paraphrased names
4. **Multiple sources** - if information comes from multiple documents, cite all of them

### Final Answer Structure
Your final answer should:
1. **Start with a direct answer** to the user's question
2. **Provide details** with inline citations
3. **End with a Sources section** listing all documents consulted:
```
## Sources Consulted
- 01_master_agreement.pdf - Main acquisition terms
- 10_stock_purchase.pdf - Stock component details
- 09_escrow_agreement.pdf - Escrow terms and release schedule
```

You must respond with valid JSON matching the Action schema.

## RESPONSE FORMAT (JSON Schema)

You MUST respond with a JSON object with exactly two keys: "action" and "reason".

### For a tool call:
```json
{
  "action": {
    "tool_name": "scan_folder",
    "tool_input": [
      {"parameter_name": "directory", "parameter_value": "."}
    ]
  },
  "reason": "I need to scan the folder to find relevant documents."
}
```

### For navigating into a directory:
```json
{
  "action": {
    "directory": "/path/to/subdir"
  },
  "reason": "I see a relevant subfolder to explore."
}
```

### For stopping with a final answer:
```json
{
  "action": {
    "final_result": "The total purchase price is $X. [Source: file.pdf, Section Y]"
  },
  "reason": "I found the answer in the documents."
}
```

### For asking the user a clarification:
```json
{
  "action": {
    "question": "Which specific agreement are you asking about?"
  },
  "reason": "The query is ambiguous."
}
```

CRITICAL: Always respond with the FULL JSON object structure shown above. NEVER respond with just a tool name or a plain string.
"""
from typing import Any, Optional, Union, List, Tuple, Dict
class FsExplorerAgent:
    """AI agent for exploring filesystems using LangChain with DeepSeek."""
    # Avant : def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        """
        Initialize the agent with DeepSeek credentials.

        Args:
            api_key: DeepSeek API key (or DEEPSEEK_API_KEY env var)
            base_url: DeepSeek base URL (default: https://api.deepseek.com)
        """
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError(
                "DEEPSEEK_API_KEY not found in environment. "
                "Please set it or provide it to the constructor."
            )

        if base_url is None:
            base_url = "https://api.deepseek.com"

        # Initialize DeepSeek via OpenAI-compatible API
        self._llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        self._messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
        self.token_usage = TokenUsage()

    def configure_task(self, task: str) -> None:
        """Add a task message to the conversation."""
        self._messages.append(HumanMessage(content=task))
    async def take_action(self) -> Optional[Tuple[Action, str]]:
        """
        Request the next action from the AI model.

        Returns:
            A tuple of (Action, ActionType) if successful, None otherwise.
        """
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Call LLM
                response = await self._llm.ainvoke(self._messages)

                # Track usage
                if hasattr(response, 'response_metadata'):
                    usage = response.response_metadata.get('token_usage', {})
                    self.token_usage.add_api_call(
                        prompt_tokens=usage.get('prompt_tokens', 0),
                        completion_tokens=usage.get('completion_tokens', 0),
                    )

                # Parse response
                self._messages.append(response)

                # Parse JSON action
                action = Action.model_validate_json(response.content)
                action_type = action.to_action_type()

                # Execute tool if needed
                if action_type == "toolcall":
                    from models import ToolCallAction
                    toolcall = action.action
                    if isinstance(toolcall, ToolCallAction):
                        self.call_tool(
                            tool_name=toolcall.tool_name,
                            tool_input=toolcall.to_fn_args(),
                        )

                return action, action_type

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Feed the error back so the LLM can self-correct
                    example = '{"action": {"tool_name": "scan_folder", "tool_input": [{"parameter_name": "directory", "parameter_value": "."}]}, "reason": "..."}'

                    error_feedback = (

                        "Your previous response was not valid JSON. Error: " + str(e) + "\n\n"

                        "You MUST respond with a JSON object like:\n" + example

                    )

                    self._messages.append(
                        HumanMessage(content=error_feedback)
                    )
                    print(f"Retry {attempt + 1}/{max_retries}: LLM returned invalid JSON, sending correction...")
                    continue

        print(f"Error in take_action after {max_retries + 1} attempts: {last_error}")
        return None

    def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        """Execute a tool and add the result to conversation."""
        try:
            result = TOOLS[tool_name](**tool_input)
        except Exception as e:
            result = f"Error calling tool {tool_name} with {tool_input}: {e}"

        self.token_usage.add_tool_result(result, tool_name)

        self._messages.append(
            HumanMessage(content=f"Tool result for {tool_name}:\n\n{result}")
        )

    def reset(self) -> None:
        """Reset the agent's conversation history."""
        self._messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self.token_usage = TokenUsage()