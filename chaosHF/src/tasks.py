from typing import Dict, Any
from .models import State, Reward

STEP_REWARD_CAP = 0.15

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
            "expected_test": 0.68,      # score BEFORE fix
            "corrected_test": 0.79      # score AFTER fix
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
            "expected_test_leaked": 0.45,   # score if leakage NOT fixed
            "expected_test_fixed": 0.72     # score if leakage IS fixed
        }
    }
}


def grade_pipeline(state: State, task_config: Dict) -> Reward:
    gt = task_config["ground_truth"]
    max_steps = task_config["max_steps"]
    task_name = task_config["name"]

    # ── Step-wise micro-rewards (strictly capped) ──────────────────────────
    # These represent observable progress milestones, not the final reward.
    # They cannot substitute for a genuine confirmed test_score.
    step_reward = 0.0
    if state.model_status == "loaded":
        step_reward += 0.02
    if state.validation_score is not None:
        step_reward += 0.03
    if state.pipeline_valid:
        step_reward += 0.03
    step_reward = min(step_reward, STEP_REWARD_CAP)

    # ── No confirmed test_score yet: return only the capped micro-reward ───
    if state.test_score is None:
        penalty = 0.05 * max(0, state.consecutive_repeats - 1)
        total = max(0.0, round(step_reward - penalty, 2))
        return Reward(
            total=total,
            pipeline_score=round(step_reward, 2),
            generalization_score=0.0,
            efficiency_score=0.0,
            penalty=round(penalty, 2),
            info="Awaiting confirmed evaluation"
        )

    # ── From here: test_score exists ───────────────────────────────────────

    # 1. Generalization (70% of total) — primary signal, strictly test_score-based
    if task_name == "easy":
        gen = (state.test_score / gt["expected_test"]) * 0.70
    elif task_name == "medium":
        target = gt["corrected_test"] if state.pipeline_valid else gt["expected_test"]
        gen = (state.test_score / target) * 0.70
    else:  # hard
        target = (
            gt["expected_test_fixed"]
            if state.pipeline_valid
            else gt["expected_test_leaked"]
        )
        gen = (state.test_score / target) * 0.70
    gen = min(0.70, max(0.0, gen))

    # 2. Efficiency bonus (15%) — rewards completing the task in fewer steps
    step_ratio = state.step_count / max_steps
    efficiency = max(0.0, 0.15 * (1.0 - step_ratio))

    # 3. Pipeline integrity bonus (15%) — all required stages completed in order
    integrity = 0.15 if state.pipeline_valid else 0.0

    # ── Penalties ──────────────────────────────────────────────────────────
    penalty = 0.0

    # Trivial policy: episode ended in fewer than 3 meaningful steps
    if state.step_count < 3:
        penalty += 0.30

    # Premature / invalid evaluation: evaluated without fixing pipeline
    if not state.pipeline_valid and state.test_score is not None:
        penalty += 0.20

    # Repeated actions
    if state.consecutive_repeats >= 2:
        penalty += 0.05 * state.consecutive_repeats

    # Overfitting signal: large val/test gap without having fixed a known issue
    if state.validation_score is not None and state.test_score is not None:
        gap = abs(state.validation_score - state.test_score)
        if gap > 0.20:
            penalty += 0.05 * (gap - 0.20)

    total = min(1.0, max(0.0, gen + efficiency + integrity - penalty))

    return Reward(
        total=round(total, 2),
        pipeline_score=round(integrity, 2),
        generalization_score=round(gen, 2),
        efficiency_score=round(efficiency, 2),
        penalty=round(penalty, 2),
        info=(
            f"test={state.test_score} | steps={state.step_count}/{max_steps}"
            f" | valid={state.pipeline_valid} | task={task_name}"
        )
    )