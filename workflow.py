"""
Workflow orchestration using LangGraph.
Version corrigée pour Python 3.9
"""
import os
from typing import Annotated, TypedDict, Literal, Optional, Union, List, Dict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from agent import FsExplorerAgent
from models import GoDeeperAction, ToolCallAction, StopAction, AskHumanAction
from fs import describe_dir_content

# Global agent instance
# CORRECTION: Remplacement de | None par Optional
_AGENT: Optional[FsExplorerAgent] = None

def get_agent() -> FsExplorerAgent:
    """Get or create the singleton agent instance."""
    global _AGENT
    if _AGENT is None:
        _AGENT = FsExplorerAgent()
    return _AGENT

def reset_agent() -> None:
    """Reset the agent instance."""
    global _AGENT
    _AGENT = None

# State definition
class GraphState(TypedDict):
    """State for the LangGraph workflow."""
    # Note: list[BaseMessage] est supporté en 3.9, mais List est plus sûr pour LangGraph
    messages: Annotated[List[BaseMessage], add_messages]
    initial_task: str
    current_directory: str
    final_result: Optional[str] # CORRECTION: Optional
    error: Optional[str]        # CORRECTION: Optional
    step_count: int
    events: List[Dict]          # CORRECTION: List/Dict

# Event types for streaming
class ToolCallEvent:
    def __init__(self, tool_name: str, tool_input: dict, reason: str):
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.reason = reason

class GoDeeperEvent:
    def __init__(self, directory: str, reason: str):
        self.directory = directory
        self.reason = reason

class AskHumanEvent:
    def __init__(self, question: str, reason: str):
        self.question = question
        self.reason = reason

class ExplorationEndEvent:
    # CORRECTION: Optional
    def __init__(self, final_result: Optional[str] = None, error: Optional[str] = None):
        self.final_result = final_result
        self.error = error

# Node functions
async def start_exploration(state: GraphState) -> GraphState:
    """Initialize exploration with the user's task."""
    agent = get_agent()

    current_dir = state.get("current_directory", ".")
    task = state["initial_task"]

    dir_description = describe_dir_content(current_dir)
    agent.configure_task(
        f"Given that the current directory ('{current_dir}') looks like this:\n\n"
        f"```text\n{dir_description}\n```\n\n"
        f"And that the user is giving you this task: '{task}', "
        f"what action should you take first?"
    )

    result = await agent.take_action()
    if result is None:
        return {**state, "error": "Could not produce action to take"}

    action, action_type = result

    # Record event
    event = None
    if action_type == "toolcall":
        toolcall = action.action
        if isinstance(toolcall, ToolCallAction):
            event = {
                "type": "tool_call",
                "step": state.get("step_count", 0) + 1,
                "tool_name": toolcall.tool_name,
                "tool_input": toolcall.to_fn_args(),
                "reason": action.reason
            }
    elif action_type == "godeeper":
        godeeper = action.action
        if isinstance(godeeper, GoDeeperAction):
            event = {
                "type": "go_deeper",
                "step": state.get("step_count", 0) + 1,
                "directory": godeeper.directory,
                "reason": action.reason
            }
            state["current_directory"] = godeeper.directory
    elif action_type == "askhuman":
        askhuman = action.action
        if isinstance(askhuman, AskHumanAction):
            event = {
                "type": "ask_human",
                "step": state.get("step_count", 0) + 1,
                "question": askhuman.question,
                "reason": action.reason
            }
    elif action_type == "stop":
        stopaction = action.action
        if isinstance(stopaction, StopAction):
            return {
                **state,
                "final_result": stopaction.final_result,
                "step_count": state.get("step_count", 0) + 1
            }

    events = state.get("events", [])
    if event:
        events.append(event)

    return {
        **state,
        "events": events,
        "step_count": state.get("step_count", 0) + 1,
        "next_action": action_type
    }

async def continue_exploration(state: GraphState) -> GraphState:
    """Continue exploration after a tool call or navigation."""
    agent = get_agent()

    current_dir = state.get("current_directory", ".")
    task = state["initial_task"]

    # Check if we need to update context
    last_event = state.get("events", [])[-1] if state.get("events") else None

    if last_event and last_event.get("type") == "go_deeper":
        dir_description = describe_dir_content(current_dir)
        agent.configure_task(
            f"Given that the current directory ('{current_dir}') "
            f"looks like this:\n\n```text\n{dir_description}\n```\n\n"
            f"And that the user is giving you this task: '{task}', "
            f"what action should you take next?"
        )
    else:
        agent.configure_task(
            "Given the result from the tool call you just performed, "
            "what action should you take next?"
        )

    result = await agent.take_action()
    if result is None:
        return {**state, "error": "Could not produce action to take"}

    action, action_type = result

    # Record event
    event = None
    if action_type == "toolcall":
        toolcall = action.action
        if isinstance(toolcall, ToolCallAction):
            event = {
                "type": "tool_call",
                "step": state.get("step_count", 0) + 1,
                "tool_name": toolcall.tool_name,
                "tool_input": toolcall.to_fn_args(),
                "reason": action.reason
            }
    elif action_type == "godeeper":
        godeeper = action.action
        if isinstance(godeeper, GoDeeperAction):
            event = {
                "type": "go_deeper",
                "step": state.get("step_count", 0) + 1,
                "directory": godeeper.directory,
                "reason": action.reason
            }
            state["current_directory"] = godeeper.directory
    elif action_type == "askhuman":
        askhuman = action.action
        if isinstance(askhuman, AskHumanAction):
            event = {
                "type": "ask_human",
                "step": state.get("step_count", 0) + 1,
                "question": askhuman.question,
                "reason": action.reason
            }
    elif action_type == "stop":
        stopaction = action.action
        if isinstance(stopaction, StopAction):
            return {
                **state,
                "final_result": stopaction.final_result,
                "step_count": state.get("step_count", 0) + 1
            }

    events = state.get("events", [])
    if event:
        events.append(event)

    return {
        **state,
        "events": events,
        "step_count": state.get("step_count", 0) + 1,
        "next_action": action_type
    }

def should_continue(state: GraphState) -> Literal["continue", "end"]:
    """Decide whether to continue or end the workflow."""
    if state.get("final_result") or state.get("error"):
        return "end"

    next_action = state.get("next_action")
    if next_action == "stop":
        return "end"

    if next_action == "askhuman":
        return "end"

    return "continue"

# Build the graph
def create_workflow() -> StateGraph:
    """Create the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("start", start_exploration)
    workflow.add_node("continue", continue_exploration)

    # Add edges
    workflow.set_entry_point("start")
    workflow.add_conditional_edges(
        "start",
        should_continue,
        {
            "continue": "continue",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "continue",
        should_continue,
        {
            "continue": "continue",
            "end": END
        }
    )

    return workflow.compile()

# Create the compiled workflow
workflow = create_workflow()