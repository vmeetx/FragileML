from typing import Dict, Any
from .models import State, Reward

# ✅ FIX: Removed all trailing spaces from keys and values. 
# ✅ FIX: Used simple names ("easy", "medium", "hard") for robust logic matching.
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
            "hint": "Check dependency versions. Fix them, then train."
        },
        "ground_truth": {
            "required_deps": {"transformers": "4.25.0", "torch": "1.13.0"},
            "expected_val": 0.85,
            "expected_test": 0.82
        }
    },
    "medium": {
        "name": "medium",
        "difficulty": 2,
        "max_steps": 12,
        "initial": {
            "dataset": {"name": "text_gen", "size": 5000, "leakage": False},
            "model": {"name": "gpt2-small", "version": "4.35.0", "status": "loaded"},
            "logs": ["Model loaded successfully", "Training completed", "Validation: 0.91"],
            "validation_score": 0.91,
            "test_score": None,
            "hint": "High validation score but outputs seem off. Check for silent failures."
        },
        "ground_truth": {
            "silent_issue": "tokenization_mismatch",
            "expected_val": 0.91,
            "expected_test": 0.68, # The score BEFORE fix
            "corrected_test": 0.79 # The score AFTER fix
        }
    },
    "hard": {
        "name": "hard",
        "difficulty": 3,
        "max_steps": 15,
        "initial": {
            "dataset": {"name": "tabular_reg", "size": 10000, "leakage": True},
            "model": {"name": "xgboost", "version": "1.7.0", "status": "loaded"},
            "logs": ["Data loaded", "Train/val split: 80/20", "Validation R²: 0.98"],
            "validation_score": 0.98,
            "test_score": None,
            "hint": "Suspiciously high validation score. Check for data leakage in split."
        },
        "ground_truth": {
            "leakage_type": "temporal",
            "expected_val_leaked": 0.98,
            "expected_test_leaked": 0.45, # Score if leakage NOT fixed
            "expected_test_fixed": 0.72   # Score if leakage IS fixed
        }
    }
}

def grade_pipeline(state: State, task_config: Dict) -> Reward:
    pipeline = 0.0
    generalization = 0.0
    efficiency = 0.0
    penalty = 0.0
    gt = task_config["ground_truth"]
    max_steps = task_config["max_steps"]
    task_name = task_config["name"]

    # 1. Milestones
    if state.model_status == "loaded": pipeline += 0.15
    if state.validation_score is not None: pipeline += 0.15
    if state.pipeline_valid: pipeline += 0.15
    if state.test_score is not None: pipeline += 0.20

    # 2. Generalization
    if state.test_score is not None:
        if task_name == "easy":
            generalization = max(0, min(1, state.test_score / gt["expected_test"])) * 0.20
        elif task_name == "medium":
            target = gt["corrected_test"] if state.pipeline_valid else gt["expected_test"]
            generalization = max(0, min(1, state.test_score / target)) * 0.20
        else:  # hard
            target = gt.get("expected_test_fixed", 0.72) if state.pipeline_valid else gt.get("expected_test_leaked", 0.45)
            if state.pipeline_valid and state.test_score >= target:
                generalization = 0.30
            elif state.test_score >= gt["expected_test_leaked"]:
                generalization = 0.10

    # 3. Efficiency
    step_ratio = state.step_count / max_steps
    efficiency = max(0.0, 0.15 - (step_ratio * 0.1))

    # 4. Penalties
    if state.consecutive_repeats >= 2: penalty += 0.05 * state.consecutive_repeats
    if state.test_score is not None and not state.pipeline_valid: penalty += 0.20
    if state.validation_score and state.test_score:
        gap = abs(state.validation_score - state.test_score)
        if gap > 0.2: penalty += 0.05 * (gap - 0.2)

    total = min(1.0, max(0.0, pipeline + generalization + efficiency - penalty))

    return Reward(
        total=round(total, 2),
        pipeline_score=round(pipeline, 2),
        generalization_score=round(generalization, 2),
        efficiency_score=round(efficiency, 2),
        penalty=round(penalty, 2),
        info=f"Stage: {state.model_status} | Val: {state.validation_score} | Test: {state.test_score}"
    )