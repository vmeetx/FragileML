from typing import Dict, List, Any
from .models import Action, Reward, ActionType, State

TASKS = {
    "easy": {
        "name": "version_hell",
        "difficulty": 1,
        "max_steps": 8,
        "initial": {
            "dataset": {"name": "synthetic_cls", "size": 1000, "leakage": False},
            "model": {"name": "bert-base", "version": "4.30.0", "status": "failed"},
            "logs": ["Error: transformers==4.25.0 required, found 4.30.0", "Model load failed"],
            "validation_score": None,
            "test_score": None,
            "hint": "Check dependency versions. Model config may need adjustment."
        },
        "ground_truth": {
            "required_deps": {"transformers": "4.25.0", "torch": "1.13.0"},
            "correct_config": {"model": "bert-base", "max_length": 128},
            "expected_val": 0.85,
            "expected_test": 0.82
        }
    },
    "medium": {
        "name": "silent_failure",
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
            "fix_action": "preprocess_data",
            "expected_val": 0.91,
            "expected_test": 0.68,
            "corrected_test": 0.79
        }
    },
    "hard": {
        "name": "data_leakage_trap",
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
            "fix_action": "split_data",
            "split_method": "time_series",
            "expected_val_leaked": 0.98,
            "expected_test_leaked": 0.45,
            "expected_test_fixed": 0.72
        }
    }
}

def grade_pipeline(state: State, task_config: Dict) -> Reward:
    pipeline = 0.0
    generalization = 0.0
    efficiency = 0.0
    penalty = 0.0
    
    gt = task_config["ground_truth"]
    
    # Pipeline validity scoring
    if state.pipeline_valid:
        pipeline += 0.4
    if state.model_status == "loaded" and state.validation_score is not None:
        pipeline += 0.2
    if state.test_score is not None:
        pipeline += 0.2
    
    # Generalization scoring (test score is key)
    if state.test_score is not None:
        if task_config["name"] == "easy":
            target = gt["expected_test"]
            generalization = max(0, min(1, state.test_score / target)) * 0.3
        elif task_config["name"] == "medium":
            target = gt["corrected_test"] if state.leakage_detected else gt["expected_test"]
            generalization = max(0, min(1, state.test_score / target)) * 0.3
        else:  # hard
            if state.leakage_detected and state.test_score >= gt["expected_test_fixed"]:
                generalization = 0.35
            elif state.test_score >= gt["expected_test_leaked"]:
                generalization = 0.1  # Partial credit for any test eval
            else:
                generalization = 0.0
    
    # Efficiency: fewer steps = better, but don't penalize necessary work
    max_steps = task_config["max_steps"]
    if state.step_count <= max_steps * 0.7:
        efficiency = 0.15
    elif state.step_count <= max_steps:
        efficiency = 0.1
    
    # Penalties
    if state.validation_score and state.test_score:
        gap = abs(state.validation_score - state.test_score)
        if gap > 0.3:
            penalty += 0.15 * (gap - 0.3)
    
    if len(state.actions_taken) > max_steps:
        penalty += 0.05 * (len(state.actions_taken) - max_steps)
    
    if state.model_status == "failed" and state.step_count > max_steps * 0.5:
        penalty += 0.1
    
    total = pipeline + generalization + efficiency - penalty
    total = min(1.0, max(0.0, total))
    
    return Reward(
        total=round(total, 2),
        pipeline_score=round(pipeline, 2),
        generalization_score=round(generalization, 2),
        efficiency_score=round(efficiency, 2),
        penalty=round(penalty, 2),
        info=f"Graded {task_config['name']}"
    )