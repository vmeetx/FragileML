import os
import sys
import json
import re
from typing import List

from openai import OpenAI

from src.environment import MLPipelineEnv
from src.models import Action, ActionType, Observation

API_BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK      = os.getenv("BENCHMARK", "ml-pipeline-env")
MAX_STEPS      = int(os.getenv("MAX_STEPS", "15"))

# Number of consecutive identical test_score readings required before
# the score is considered confirmed and the episode may end.
CONFIRM_STREAK = 2

SYSTEM_PROMPT = """You are an ML Pipeline Debugger. Output ONLY valid JSON.

RULES (follow in strict order — do not skip steps):
1. If model_status is 'failed' and logs mention 'version': use fix_dependency with {"transformers": "4.25.0"}.
2. If model_status is 'loaded' and validation_score is None: use train_model with {}.
3. If task is 'medium' and pipeline_valid is false: use preprocess_data with {"tokenization": true}.
4. If task is 'hard' and leakage_detected is true: use split_data with {"method": "time_series"}.
5. If pipeline_valid is true AND train_model has been used AND test_score is null: use evaluate with {"metric": "test"}.
6. If test_score is not null AND score is confirmed (you have seen it twice): output {"action_type":"done","config":{},"done":true}.
7. If test_score is not null but NOT yet confirmed: use evaluate again with {"metric": "test"} to get a second reading.
8. NEVER repeat the last action shown in History.
9. NEVER output done before test_score is confirmed.

JSON format: {"action_type":"...", "config":{}, "done":false}"""


def parse_action(text: str) -> Action:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return Action(**json.loads(match.group(0)))
    except Exception:
        return Action(action_type=ActionType.INSPECT_LOGS, config={"reason": "parse_error"}, done=False)


def build_prompt(task: str, obs: Observation, confirmed: bool) -> str:
    last_action = obs.history[-1] if obs.history else "none"
    recent_logs = " | ".join(obs.logs[-3:]) if obs.logs else "No logs"

    # Signal that the score is confirmed and the agent should end the episode
    if obs.test_score is not None and confirmed:
        return (
            f"Task: {task}\n"
            f"*** TEST SCORE = {obs.test_score} — CONFIRMED (seen {CONFIRM_STREAK}x). PIPELINE COMPLETE. ***\n"
            f'Output {{"action_type":"done","config":{{}},"done":true}} and nothing else.'
        )

    # Signal that a score exists but is not yet confirmed
    if obs.test_score is not None and not confirmed:
        return (
            f"Task: {task}\n"
            f"test_score={obs.test_score} seen once — needs one more consistent reading.\n"
            f'Use evaluate again: {{"action_type":"evaluate","config":{{"metric":"test"}},"done":false}}'
        )

    return (
        f"Task: {task} | Steps remaining: {obs.steps_remaining}\n"
        f"model_status: {obs.model_status}\n"
        f"validation_score: {obs.validation_score}\n"
        f"test_score: {obs.test_score}\n"
        f"pipeline_valid: {obs.pipeline_valid}\n"
        f"leakage_detected: {obs.leakage_detected}\n"
        f"Logs: {recent_logs}\n"
        f"History: {obs.history}\n"
        f"Last action: {last_action} — do NOT repeat it.\n"
        f"Hint: {obs.hint}\n"
        f"Choose NEXT action (JSON ONLY):"
    )


def score_is_confirmed(score: float, streak: int, evaluate_seen: bool) -> bool:
    """
    A score is trusted only when:
    - It is at or above the minimum acceptable threshold (0.70)
    - It has been observed consistently for at least CONFIRM_STREAK steps
    - At least one valid evaluate action has been recorded in the episode
    """
    return score >= 0.70 and streak >= CONFIRM_STREAK and evaluate_seen


def run_task(task: str) -> dict:
    env = MLPipelineEnv(task_name=task)
    client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)
    obs = env.reset()

    rewards: List[float] = []
    steps = 0
    success = False
    good_streak = 0
    last_score = None
    evaluate_seen = False   # True once a valid evaluate action has been executed

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")

    try:
        while steps < MAX_STEPS:

            # Track whether a successful evaluate has been recorded
            if "evaluate" in obs.history:
                evaluate_seen = True

            # ── Score stability tracking ───────────────────────────────────
            current_score = obs.test_score
            if current_score is not None and current_score >= 0.70:
                if current_score == last_score:
                    good_streak += 1
                else:
                    good_streak = 1
                    print(f"[INFO] test_score={current_score} (streak reset)")
            else:
                good_streak = 0
            last_score = current_score

            confirmed = score_is_confirmed(current_score or 0.0, good_streak, evaluate_seen)

            if confirmed:
                success = True
                print(
                    f"[INFO] Score {current_score} confirmed: "
                    f"streak={good_streak}/{CONFIRM_STREAK}, evaluate_seen={evaluate_seen}. Done."
                )
                break

            if current_score is not None and not confirmed:
                print(
                    f"[INFO] Confirming score={current_score} "
                    f"(streak={good_streak}/{CONFIRM_STREAK})"
                )

            # ── Build prompt ───────────────────────────────────────────────
            prompt = build_prompt(task, obs, confirmed)

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            action = parse_action(resp.choices[0].message.content)

            # ── Hard gate: block "done" before score is confirmed ──────────
            # If the LLM tries to end the episode prematurely, override it
            # with another evaluate call to collect the required confirmation.
            if (action.done or action.action_type == ActionType.DONE) and not confirmed:
                print(
                    f"[WARN] LLM attempted early termination "
                    f"(score={current_score}, streak={good_streak}/{CONFIRM_STREAK}). "
                    f"Overriding with evaluate."
                )
                action = Action(
                    action_type=ActionType.EVALUATE,
                    config={"metric": "test"},
                    done=False
                )

            obs, reward, done, info = env.step(action)
            rewards.append(reward.total)
            steps += 1

            err = f'"{info.get("error")}"' if info.get("error") else "null"
            print(
                f"[STEP] step={steps} action={action.model_dump_json()} "
                f"reward={reward.total:.2f} done={str(done).lower()} error={err}"
            )

            # Bail out early if reward collapses — avoids burning API credits
            if reward.total <= 0.0 and steps > 3:
                print("[WARN] Reward hit 0.00. Breaking to save credits.")
                break

            if done:
                # Episode terminated via env rules (max_steps or action.done=True)
                # Success only if the score was genuinely confirmed
                success = confirmed or score_is_confirmed(
                    obs.test_score or 0.0, good_streak, evaluate_seen
                )
                break

    except Exception as e:
        err_str = str(e)
        if "402" in err_str or "credits" in err_str.lower():
            print("[WARN] API credit limit reached. Terminating gracefully.")
        else:
            print(f"[ERROR] Unexpected failure: {err_str}")
            rewards.append(0.0)
        success = False
    finally:
        score = rewards[-1] if rewards else 0.0
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.2f} rewards={rewards_str}"
        )
        env.close()

    return {"task": task, "success": success, "score": score}


if __name__ == "__main__":
    results = [run_task(t) for t in ["easy", "medium", "hard"]]
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n# Baseline: avg={avg:.2f}", file=sys.stderr)