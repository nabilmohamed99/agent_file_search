"""
CLI entry point for the FsExplorer agent with LangGraph.
Version corrigée pour Python 3.9
"""
import asyncio
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typer import Typer, Option
from typing import Annotated, Optional, Dict, Tuple # Ajouté Optional, Dict, Tuple
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.text import Text

from workflow import workflow, get_agent, reset_agent, GraphState

app = Typer()

TOOL_ICONS = {
    "scan_folder": "📂",
    "preview_file": "👁️",
    "parse_file": "📖",
    "read": "📄",
    "grep": "🔍",
    "glob": "🔎",
}

PHASE_DESCRIPTIONS = {
    "scan_folder": ("Phase 1", "Parallel Document Scan", "cyan"),
    "preview_file": ("Phase 1/2", "Quick Preview", "cyan"),
    "parse_file": ("Phase 2", "Deep Dive", "green"),
    "read": ("Reading", "Text File", "blue"),
    "grep": ("Searching", "Pattern Match", "yellow"),
    "glob": ("Finding", "File Search", "yellow"),
}

def format_tool_panel(event: dict, step_number: int) -> Panel:
    tool_name = event["tool_name"]
    icon = TOOL_ICONS.get(tool_name, "🔧")
    phase_info = PHASE_DESCRIPTIONS.get(tool_name, ("Action", "Tool Call", "yellow"))
    phase_label, phase_desc, color = phase_info

    lines = []
    tool_input = event["tool_input"]
    if "directory" in tool_input:
        target = tool_input["directory"]
        lines.append(f"**Target Directory:** `{target}`")
    elif "file_path" in tool_input:
        target = tool_input["file_path"]
        lines.append(f"**Target File:** `{target}`")

    other_params = {k: v for k, v in tool_input.items() if k not in ("directory", "file_path")}
    if other_params:
        lines.append(f"**Parameters:** `{other_params}`")

    lines.append("\n---\n\n**Agent's Reasoning:**\n")
    lines.append(event["reason"])

    content = "\n".join(lines)
    title = f"{icon} Step {step_number}: {tool_name} [{phase_label}: {phase_desc}]"

    return Panel(Markdown(content), title=title, title_align="left", border_style=f"bold {color}", padding=(1, 2))

def format_navigation_panel(event: dict, step_number: int) -> Panel:
    content = f"**Navigating to:** `{event['directory']}`\n\n---\n\n**Agent's Reasoning:**\n\n{event['reason']}"
    return Panel(Markdown(content), title=f"📁 Step {step_number}: Navigate", title_align="left", border_style="bold magenta", padding=(1, 2))

def print_workflow_header(console: Console, task: str) -> None:
    console.print()
    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan", justify="right")
    header.add_column()
    header.add_row("🤖 FsExplorer Agent", "")
    header.add_row("📋 Task:", task)
    header.add_row("🕐 Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    console.print(Panel(header, border_style="bold blue", title="Starting Exploration"))

def print_workflow_summary(console: Console, agent, step_count: int) -> None:
    usage = agent.token_usage
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold", justify="right")
    summary.add_column()
    summary.add_row("Total Steps:", str(step_count))
    summary.add_row("API Calls:", str(usage.api_calls))
    summary.add_row("Prompt Tokens:", f"{usage.prompt_tokens:,}")
    summary.add_row("Completion Tokens:", f"{usage.completion_tokens:,}")

    input_cost, output_cost, total_cost = usage._calculate_cost()
    summary.add_row("Est. Total Cost:", f"${total_cost:.4f}")

    console.print(Panel(summary, title="📊 Workflow Summary", border_style="bold blue"))

async def run_workflow(task: str) -> None:
    console = Console()
    reset_agent()
    print_workflow_header(console, task)

    initial_state: GraphState = {
        "messages": [],
        "initial_task": task,
        "current_directory": ".",
        "final_result": None,
        "error": None,
        "step_count": 0,
        "events": []
    }

    with console.status(status="[bold cyan]🔄 Analyzing task...") as status:
        final_state = await workflow.ainvoke(initial_state)
        for event in final_state.get("events", []):
            if event["type"] == "tool_call":
                status.update(f"[bold cyan]🔧 Executing {event['tool_name']}...")
                console.print(format_tool_panel(event, event["step"]))
            elif event["type"] == "go_deeper":
                status.update("[bold cyan]🔄 Exploring directory...")
                console.print(format_navigation_panel(event, event["step"]))
        status.stop()

    if final_state.get("final_result"):
        console.print(Panel(Markdown(final_state["final_result"]), title="✅ Final Answer", border_style="bold green"))
    elif final_state.get("error"):
        console.print(Panel(Text(final_state["error"], style="bold red"), title="❌ Error", border_style="bold red"))

    print_workflow_summary(console, get_agent(), final_state.get("step_count", 0))

@app.command()
def main(
    task: Annotated[str, Option("--task", "-t", help="Task for the agent")]
) -> None:
    asyncio.run(run_workflow(task))

# POINT D'ENTRÉE INDISPENSABLE :
if __name__ == "__main__":
    app()