<<<<<<< HEAD
from typing import Dict, Any, List
from .models import State, Reward, ActionType

STEP_REWARD_CAP = 0.05
=======
from typing import Dict, Any
from .models import State, Reward

STEP_REWARD_CAP = 0.15
>>>>>>> 7cd7cb628434b7f9838c9ff95925d3d267655bc2

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
<<<<<<< HEAD
            "hint": "Fix dependency version mismatch before loading model."
=======
            "hint": "Check dependency versions. Fix them, then train."
>>>>>>> 7cd7cb628434b7f9838c9ff95925d3d267655bc2
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
<<<<<<< HEAD
            "expected_test_broken": 0.68,
            "expected_test_fixed": 0.79,
            "required_sequence": ["preprocess_data", "evaluate"]
=======
            "expected_test": 0.68,      # score BEFORE fix
            "corrected_test": 0.79      # score AFTER fix
>>>>>>> 7cd7cb628434b7f9838c9ff95925d3d267655bc2
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
<<<<<<< HEAD
            "expected_test_leaked": 0.45,
            "expected_test_fixed": 0.72,
            "required_sequence": ["split_data", "train_model", "evaluate"]
=======
            "expected_test_leaked": 0.45,   # score if leakage NOT fixed
            "expected_test_fixed": 0.72     # score if leakage IS fixed
>>>>>>> 7cd7cb628434b7f9838c9ff95925d3d267655bc2
        }
    }
}


<<<<<<< HEAD
def grade_pipeline(state: State, task_config: Dict, last_action: str = None) -> Reward:
    gt = task_config["ground_truth"]
    task_name = task_config["name"]
    
    # ── MICRO-REWARDS FOR OBSERVABLE PROGRESS (capped) ──
    micro = 0.0
    actions = [a.action_type.value for a in state.actions_taken]
    
    if "fix_dependency" in actions and state.model_status == "loaded":
        micro += 0.02
    if state.validation_score is not None:
        micro += 0.02
    if state.pipeline_valid:
        micro += 0.01
    micro = min(micro, STEP_REWARD_CAP)
    
    # ── NO CONFIRMED TEST_SCORE: return micro-reward only ──
    if state.test_score is None:
        penalty = _compute_penalties(state, task_config, last_action)
        total = max(0.0, round(micro - penalty, 2))
        return Reward(
            total=total,
            pipeline_score=round(micro, 2),
=======
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
>>>>>>> 7cd7cb628434b7f9838c9ff95925d3d267655bc2
            generalization_score=0.0,
            efficiency_score=0.0,
            penalty=round(penalty, 2),
            info="Awaiting confirmed evaluation"
        )
<<<<<<< HEAD
    
    # ── TEST_SCORE EXISTS: sparse final reward (80% from correctness) ──
    if task_name == "easy":
        target = gt["expected_test"]
        correct = abs(state.test_score - target) < 0.05
        gen = 0.80 if correct else 0.0
    elif task_name == "medium":
        if state.pipeline_valid:
            target = gt["expected_test_fixed"]
        else:
            target = gt["expected_test_broken"]
        correct = abs(state.test_score - target) < 0.05 and state.pipeline_valid
        gen = 0.80 if correct else 0.0
    else:  # hard
        if state.pipeline_valid and not state.leakage_detected:
            target = gt["expected_test_fixed"]
        else:
            target = gt["expected_test_leaked"]
        correct = (abs(state.test_score - target) < 0.05 
                   and state.pipeline_valid 
                   and not state.leakage_detected)
        gen = 0.80 if correct else 0.0
    
    # Efficiency (10%) — only if completed in reasonable steps
    min_steps = len(gt["required_sequence"])
    efficiency = 0.10 if (state.step_count <= min_steps + 2 and state.pipeline_valid) else 0.0
    
    # Sequence integrity (10%)
    sequence_ok = _check_required_sequence(state, task_config)
    integrity = 0.10 if sequence_ok else 0.0
    
    # Penalties
    penalty = _compute_penalties(state, task_config, last_action)
    if state.episode_done and not state.pipeline_valid:
        penalty += 0.30
    
    total = min(1.0, max(0.0, gen + efficiency + integrity - penalty))
    
=======

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

>>>>>>> 7cd7cb628434b7f9838c9ff95925d3d267655bc2
    return Reward(
        total=round(total, 2),
        pipeline_score=round(integrity, 2),
        generalization_score=round(gen, 2),
        efficiency_score=round(efficiency, 2),
        penalty=round(penalty, 2),
<<<<<<< HEAD
        info=f"test={state.test_score} | valid={state.pipeline_valid} | seq_ok={sequence_ok}"
    )


def _compute_penalties(state: State, task_config: Dict, last_action: str = None) -> float:
    penalty = 0.0
    if state.consecutive_repeats >= 2:
        penalty += 0.40
    if last_action == "evaluate" and not state.pipeline_valid:
        penalty += 0.50
    if last_action == "evaluate":
        task_name = task_config["name"]
        actions = [a.action_type.value for a in state.actions_taken]
        if task_name == "easy" and "train_model" not in actions:
            penalty += 0.50
        elif task_name == "hard" and not state.dataset_config.get("leakage_fixed", False):
            penalty += 0.50
    return penalty


def _check_required_sequence(state: State, task_config: Dict) -> bool:
    gt = task_config["ground_truth"]
    required = gt.get("required_sequence", [])
    if not required:
        return True
    actions = [a.action_type.value for a in state.actions_taken 
               if a.action_type.value in required + ["done"]]
    req_idx = 0
    for act in actions:
        if req_idx < len(required) and act == required[req_idx]:
            req_idx += 1
    return req_idx == len(required)
=======
        info=(
            f"test={state.test_score} | steps={state.step_count}/{max_steps}"
            f" | valid={state.pipeline_valid} | task={task_name}"
        )
    )
>>>>>>> 7cd7cb628434b7f9838c9ff95925d3d267655bc2
