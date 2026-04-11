from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ✅ FIX: Removed all trailing spaces from enum values
class ActionType(str, Enum):
    LOAD_MODEL = "load_model"
    FIX_DEPENDENCY = "fix_dependency"
    PREPROCESS_DATA = "preprocess_data"
    SPLIT_DATA = "split_data"
    TRAIN_MODEL = "train_model"
    EVALUATE = "evaluate"
    INSPECT_LOGS = "inspect_logs"
    VALIDATE_DATA = "validate_data"
    DONE = "done"

class Action(BaseModel):
    action_type: ActionType
    config: Dict[str, Any] = Field(default_factory=dict)
    done: bool = False

class Observation(BaseModel):
    dataset_summary: Dict[str, Any] = Field(default_factory=dict)
    model_status: str = "not_loaded"
    logs: List[str] = Field(default_factory=list)
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    history: List[str] = Field(default_factory=list)
    step_count: int = 0
    available_actions: List[str] = Field(default_factory=list)
    hint: Optional[str] = None
    pipeline_valid: bool = False
    leakage_detected: bool = False
    steps_remaining: int = 0

class Reward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    pipeline_score: float = 0.0
    generalization_score: float = 0.0
    efficiency_score: float = 0.0
    penalty: float = 0.0
    info: str = ""

class State(BaseModel):
    task_name: str
    step_count: int
    max_steps: int
    dataset_config: Dict[str, Any]
    model_params: Dict[str, Any]
    model_status: str
    logs: List[str]
    validation_score: Optional[float]
    test_score: Optional[float]
    pipeline_valid: bool
    actions_taken: List[Action]
    episode_done: bool
    leakage_detected: bool
    overfitting_penalty: float
    last_action_type: Optional[str] = None
    consecutive_repeats: int = 0