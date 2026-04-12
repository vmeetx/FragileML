from typing import Dict, Any, List
from .models import State, Reward, ActionType

STEP_REWARD_CAP = 0.03  # Reduced from 0.05

TASKS = {
    "easy": {
        "name": "easy",
        "difficulty": 1,
        "max_steps": 8,
        "initial": {
            "dataset": {"name": "synthetic_cls", "size": 1000, "leakage": False},
            "model": {"name": "bert-base", "version": "4.30.0", "status": "failed"},
            "logs": ["Error: transformers==4.25.0 required, found 4.30.0", "Model load failed"],
            "validation_score": None,
            "test_score": None,
            "hint": "Fix dependency version mismatch before loading model."
        },
        "ground_truth": {
            "required_deps": {"transformers": "4.25.0", "torch": "1.13.0"},
            "expected_val": 0.85,
            "expected_test": 0.82,
            "required_sequence": ["fix_dependency", "train_model", "evaluate"]
        }
    },
    "medium": {
        "name": "medium",
        "difficulty": 2,
        "max_steps": 12,
        "initial": {
            "dataset": {"name": "text_gen", "size": 5000, "leakage": False},
            "model": {"name": "gpt2-small", "version": "4.35.0", "status": "loaded"},
            "logs": ["Model loaded", "Validation: 0.91", "⚠️ Outputs inconsistent with validation"],
            "validation_score": 0.91,
            "test_score": None,
            "hint": "High validation but poor outputs suggests tokenization mismatch."
        },
        "ground_truth": {
            "silent_issue": "tokenization_mismatch",
            "expected_val": 0.91,
            "expected_test_broken": 0.68,
            "expected_test_fixed": 0.79,
            "required_sequence": ["preprocess_data", "evaluate"]
        }
    },
    "hard": {
        "name": "hard",
        "difficulty": 3,
        "max_steps": 15,
        "initial": {
            "dataset": {"name": "tabular_reg", "size": 10000, "leakage": True},
            "model": {"name": "xgboost", "version": "1.7.0", "status": "loaded"},
            "logs": ["Validation R²: 0.98", "⚠️ Temporal ordering may cause leakage"],
            "validation_score": 0.98,
            "test_score": None,
            "hint": "Suspiciously high validation suggests temporal data leakage."
        },
        "ground_truth": {
            "leakage_type": "temporal",
            "expected_val_leaked": 0.98,
            "expected_test_leaked": 0.45,
            "expected_test_fixed": 0.72,
            "required_sequence": ["split_data", "train_model", "evaluate"]
        }
    }
}


def grade_pipeline(state: State, task_config: Dict, last_action: str = None) -> Reward:
    gt = task_config["ground_truth"]
    task_name = task_config["name"]
    actions = [a.action_type.value for a in state.actions_taken]
    
    # ── HARD GATE: Required sequence must be satisfied for ANY meaningful reward ──
    sequence_ok = _check_required_sequence(state, task_config)
    
    # ── MICRO-REWARDS FOR OBSERVABLE PROGRESS (strictly capped) ──
    micro = 0.0
    if "fix_dependency" in actions and state.model_status == "loaded":
        micro += 0.01
    if state.validation_score is not None:
        micro += 0.01
    if state.pipeline_valid:
        micro += 0.01
    micro = min(micro, STEP_REWARD_CAP)
    
    # ── NO CONFIRMED TEST_SCORE: return micro-reward only ──
    if state.test_score is None:
        penalty = _compute_penalties(state, task_config, last_action, actions)
        total = max(0.0, round(micro - penalty, 2))
        return Reward(
            total=total,
            pipeline_score=round(micro, 2),
            generalization_score=0.0,
            efficiency_score=0.0,
            penalty=round(penalty, 2),
            info="Awaiting confirmed evaluation"
        )
    
    # ── HARD GATE: If sequence not satisfied → ZERO reward regardless of test_score ──
    if not sequence_ok:
        return Reward(
            total=0.0,
            pipeline_score=0.0,
            generalization_score=0.0,
            efficiency_score=0.0,
            penalty=0.50,
            info="Required sequence not satisfied — reward blocked"
        )
    
    # ── DISCRETE REWARD: No ratios, only threshold checks ──
    if task_name == "easy":
        target = gt["expected_test"]
        correct = state.test_score >= (target - 0.05)
        gen = 0.85 if correct else 0.0
    elif task_name == "medium":
        if state.pipeline_valid:
            target = gt["expected_test_fixed"]
        else:
            target = gt["expected_test_broken"]
        correct = state.test_score >= (target - 0.05) and state.pipeline_valid
        gen = 0.85 if correct else 0.0
    else:  # hard
        if state.pipeline_valid and not state.leakage_detected:
            target = gt["expected_test_fixed"]
        else:
            target = gt["expected_test_leaked"]
        correct = (state.test_score >= (target - 0.05) 
                   and state.pipeline_valid 
                   and not state.leakage_detected)
        gen = 0.85 if correct else 0.0
    
    # Efficiency (10%) — only if completed in reasonable steps AND sequence valid
    min_steps = len(gt["required_sequence"])
    efficiency = 0.10 if (state.step_count <= min_steps + 2 and sequence_ok) else 0.0
    
    # Sequence integrity bonus (5%) — already enforced by hard gate, small bonus for exact order
    integrity = 0.05 if sequence_ok else 0.0
    
    # Penalties
    penalty = _compute_penalties(state, task_config, last_action, actions)
    
    total = min(1.0, max(0.0, gen + efficiency + integrity - penalty))
    
    return Reward(
        total=round(total, 2),
        pipeline_score=round(integrity, 2),
        generalization_score=round(gen, 2),
        efficiency_score=round(efficiency, 2),
        penalty=round(penalty, 2),
        info=f"test={state.test_score} | seq_ok={sequence_ok} | steps={state.step_count}"
    )


def _compute_penalties(state: State, task_config: Dict, last_action: str, actions: List[str]) -> float:
    penalty = 0.0
    
    # Hard punish repeated actions (≥2 consecutive same action)
    if state.consecutive_repeats >= 2:
        penalty += 0.50
    
    # Hard punish invalid evaluate spam
    if last_action == "evaluate":
        # Count recent evaluate actions
        recent_evals = sum(1 for a in actions[-3:] if a == "evaluate")
        if recent_evals >= 2:
            penalty += 0.50
        
        # Premature evaluate without required prerequisites
        task_name = task_config["name"]
        if task_name == "easy" and "train_model" not in actions:
            penalty += 0.50
        elif task_name == "hard" and not state.dataset_config.get("leakage_fixed", False):
            penalty += 0.50
        elif task_name == "medium" and not state.pipeline_valid:
            penalty += 0.50
    
    # Penalty for episode ending without valid pipeline
    if state.episode_done and not state.pipeline_valid:
        penalty += 0.30
    
    return penalty


def _check_required_sequence(state: State, task_config: Dict) -> bool:
    """Check that required actions appear in history in correct order."""
    gt = task_config["ground_truth"]
    required = gt.get("required_sequence", [])
    if not required:
        return True
    
    # Extract only relevant actions from history
    actions = [a.action_type.value for a in state.actions_taken 
               if a.action_type.value in required]
    
    # Check subsequence match (order matters, but allows interleaving)
    req_idx = 0
    for act in actions:
        if req_idx < len(required) and act == required[req_idx]:
            req_idx += 1
    
    # All required steps must have been executed in order
    return req_idx == len(required)