from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    LOAD_MODEL       = "load_model"
    FIX_DEPENDENCY   = "fix_dependency"
    PREPROCESS_DATA  = "preprocess_data"
    SPLIT_DATA       = "split_data"
    TRAIN_MODEL      = "train_model"
    EVALUATE         = "evaluate"
    INSPECT_LOGS     = "inspect_logs"
    VALIDATE_DATA    = "validate_data"
    DONE             = "done"
    # ── new actions for extended tasks ──────────────────────────────────
    INSTALL_PACKAGE  = "install_package"   # import_error: install missing dep
    CLEAR_CACHE      = "clear_cache"       # cache_corrupt: wipe stale cache
    RELOAD_MODEL     = "reload_model"      # safetensor_fail: force fresh download
    REDUCE_BATCH     = "reduce_batch"      # oom_crash: lower batch size and retry


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
    # ── extended state fields ────────────────────────────────────────────
    cache_cleared: bool = False       # cache_corrupt task
    weights_valid: bool = True        # safetensor_fail task
    batch_size: Optional[int] = None  # oom_crash task
    oom_triggered: bool = False       # oom_crash task