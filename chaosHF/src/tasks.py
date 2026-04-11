from typing import Dict, Any
from .models import State, Reward

TASKS = {
    # ── existing tasks ───────────────────────────────────────────────────────
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
            "expected_test": 0.68,
            "corrected_test": 0.79
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
            "expected_test_leaked": 0.45,
            "expected_test_fixed": 0.72
        }
    },

    # ── new tasks ────────────────────────────────────────────────────────────

    # Task 4: Missing package — agent must identify and install 'accelerate'
    "import_error": {
        "name": "import_error",
        "difficulty": 2,
        "max_steps": 10,
        "initial": {
            "dataset": {"name": "image_cls", "size": 3000, "leakage": False},
            "model": {"name": "vit-base", "version": "4.35.0", "status": "import_error"},
            "logs": [
                "Traceback (most recent call last):",
                "ModuleNotFoundError: No module named 'accelerate'",
                "Model load aborted."
            ],
            "validation_score": None,
            "test_score": None,
            "hint": "Model failed to load due to a missing package. Read the logs carefully."
        },
        "ground_truth": {
            "missing_package": "accelerate",
            "required_version": "0.21.0",
            "expected_val": 0.88,
            "expected_test": 0.84
        }
    },

    # Task 5: Cache corruption — stale cached weights cause flaky validation
    "cache_corrupt": {
        "name": "cache_corrupt",
        "difficulty": 3,
        "max_steps": 12,
        "initial": {
            "dataset": {"name": "nlp_cls", "size": 8000, "leakage": False},
            "model": {"name": "roberta-base", "version": "4.35.0", "status": "loaded"},
            "logs": [
                "Model loaded from cache.",
                "Training completed.",
                "Warning: Validation score unstable across runs (0.91 / 0.73 / 0.88)",
                "Possible cache inconsistency detected."
            ],
            "validation_score": 0.73,   # worst-case reading — looks bad but cause is cache
            "test_score": None,
            "hint": "Validation score is inconsistent across runs. Something is wrong before training."
        },
        "ground_truth": {
            "root_cause": "stale_cache",
            "expected_val_after_fix": 0.89,
            "expected_test": 0.86
        }
    },

    # Task 6: Safetensor load failure — partial weights, missing keys in state_dict
    "safetensor_fail": {
        "name": "safetensor_fail",
        "difficulty": 3,
        "max_steps": 12,
        "initial": {
            "dataset": {"name": "seq2seq", "size": 6000, "leakage": False},
            "model": {"name": "t5-small", "version": "4.35.0", "status": "partial"},
            "logs": [
                "Loading weights from safetensors...",
                "Error: Missing keys in state_dict: ['encoder.block.0.layer.0.SelfAttention.q.weight']",
                "Warning: Model loaded with 12 missing weight tensors.",
                "Inference may produce garbage outputs."
            ],
            "validation_score": None,
            "test_score": None,
            "hint": "Model loaded but weights are incomplete. Check the safetensor file integrity."
        },
        "ground_truth": {
            "missing_keys": 12,
            "fix": "force_download",
            "expected_val": 0.83,
            "expected_test": 0.80
        }
    },

    # Task 7: OOM crash — training silently dies, agent must reduce batch size
    "oom_crash": {
        "name": "oom_crash",
        "difficulty": 4,
        "max_steps": 15,
        "initial": {
            "dataset": {"name": "large_lm", "size": 50000, "leakage": False},
            "model": {"name": "llama-7b", "version": "4.35.0", "status": "loaded"},
            "logs": [
                "Training started with batch_size=32.",
                "CUDA out of memory. Tried to allocate 2.50 GiB.",
                "RuntimeError: CUDA error: device-side assert triggered.",
                "Training process killed."
            ],
            "validation_score": None,
            "test_score": None,
            "hint": "Training crashed silently. Check memory usage. The batch size may be too large."
        },
        "ground_truth": {
            "oom_batch_size": 32,
            "safe_batch_size": 8,
            "expected_val": 0.81,
            "expected_test": 0.78
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
    if state.model_status == "loaded":          pipeline += 0.15
    if state.validation_score is not None:      pipeline += 0.15
    if state.pipeline_valid:                    pipeline += 0.15
    if state.test_score is not None:            pipeline += 0.20

    # 2. Generalization — per task
    if state.test_score is not None:
        if task_name == "easy":
            generalization = max(0, min(1, state.test_score / gt["expected_test"])) * 0.20

        elif task_name == "medium":
            target = gt["corrected_test"] if state.pipeline_valid else gt["expected_test"]
            generalization = max(0, min(1, state.test_score / target)) * 0.20

        elif task_name == "hard":
            target = gt.get("expected_test_fixed", 0.72) if state.pipeline_valid \
                     else gt.get("expected_test_leaked", 0.45)
            if state.pipeline_valid and state.test_score >= target:
                generalization = 0.30
            elif state.test_score >= gt["expected_test_leaked"]:
                generalization = 0.10

        elif task_name == "import_error":
            generalization = max(0, min(1, state.test_score / gt["expected_test"])) * 0.20

        elif task_name == "cache_corrupt":
            # Full generalization only if cache was actually cleared before eval
            if state.cache_cleared:
                generalization = max(0, min(1, state.test_score / gt["expected_test"])) * 0.25
            else:
                generalization = max(0, min(1, state.test_score / gt["expected_test"])) * 0.05

        elif task_name == "safetensor_fail":
            # Full generalization only if weights were reloaded cleanly
            if state.weights_valid:
                generalization = max(0, min(1, state.test_score / gt["expected_test"])) * 0.25
            else:
                generalization = 0.0  # garbage outputs — no credit

        elif task_name == "oom_crash":
            # Full generalization only if batch was reduced before training succeeded
            if not state.oom_triggered:
                generalization = max(0, min(1, state.test_score / gt["expected_test"])) * 0.25
            else:
                generalization = max(0, min(1, state.test_score / gt["expected_test"])) * 0.10

    # 3. Efficiency
    step_ratio = state.step_count / max_steps
    efficiency = max(0.0, 0.15 - (step_ratio * 0.1))

    # 4. Penalties
    if state.consecutive_repeats >= 2:
        penalty += 0.05 * state.consecutive_repeats
    if state.test_score is not None and not state.pipeline_valid:
        penalty += 0.20
    if state.validation_score and state.test_score:
        gap = abs(state.validation_score - state.test_score)
        if gap > 0.2:
            penalty += 0.05 * (gap - 0.2)

    total = min(1.0, max(0.0, pipeline + generalization + efficiency - penalty))

    return Reward(
        total=round(total, 2),
        pipeline_score=round(pipeline, 2),
        generalization_score=round(generalization, 2),
        efficiency_score=round(efficiency, 2),
        penalty=round(penalty, 2),
        info=f"Stage: {state.model_status} | Val: {state.validation_score} | Test: {state.test_score}"
    )