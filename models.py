"""
Pydantic models for FsExplorer agent actions.
Version complète corrigée pour Python 3.9
"""
from pydantic import BaseModel, Field
from typing import Literal, Any, Union, List, Dict
from typing_extensions import TypeAlias

# TypeAlias est requis via typing_extensions pour Python < 3.10
Tools: TypeAlias = Literal[
    "read", "grep", "glob", "scan_folder", "preview_file", "parse_file"
]

ActionType: TypeAlias = Literal["stop", "godeeper", "toolcall", "askhuman"]

class StopAction(BaseModel):
    final_result: str = Field(
        description="Final result of the operation with the answer to the user's query"
    )

class AskHumanAction(BaseModel):
    question: str = Field(
        description="Clarification question to ask the user"
    )

class GoDeeperAction(BaseModel):
    directory: str = Field(
        description="Path to the directory to navigate into"
    )

class ToolCallArg(BaseModel):
    parameter_name: str = Field(description="Name of the parameter")
    parameter_value: Any = Field(description="Value for the parameter")

class ToolCallAction(BaseModel):
    tool_name: Tools = Field(description="Name of the tool to invoke")
    # Utilisation de List[] au lieu de list[] pour la compatibilité 3.9
    tool_input: List[ToolCallArg] = Field(description="Arguments to pass to the tool")

    def to_fn_args(self) -> Dict[str, Any]:
        return {arg.parameter_name: arg.parameter_value for arg in self.tool_input}

class Action(BaseModel):
    # CORRECTION : Union au lieu de l'opérateur pipe '|'
    action: Union[ToolCallAction, GoDeeperAction, StopAction, AskHumanAction] = Field(
        description="The specific action to take"
    )
    reason: str = Field(
        description="Explanation for why this action was chosen"
    )

    def to_action_type(self) -> ActionType:
        if isinstance(self.action, ToolCallAction):
            return "toolcall"
        elif isinstance(self.action, GoDeeperAction):
            return "godeeper"
        elif isinstance(self.action, AskHumanAction):
            return "askhuman"
        else:
            return "stop"