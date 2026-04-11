import os, sys, json, re
from typing import List
from openai import OpenAI
from src.environment import MLPipelineEnv
from src.models import Action, ActionType, Observation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK    = os.getenv("BENCHMARK", "ml-pipeline-env")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "15"))
CONFIRM_STREAK = 2

SYSTEM_PROMPT = """You are an ML Pipeline Debugger. Output ONLY valid JSON.
RULES:
1. If model_status is 'failed' and logs mention 'version': use fix_dependency with {"transformers": "4.25.0"}.
2. If model_status is 'loaded' and validation_score is None: use train_model.
3. If task is 'medium' and validation >= 0.9 and pipeline_valid is false: use preprocess_data with {"tokenization": true}.
4. If task is 'hard' and leakage_detected is true: use split_data with {"method": "time_series"}.
5. If pipeline_valid is true and test_score is null: use evaluate with {"metric": "test"}.
6. If test_score is not null: output {"action_type":"done","config":{},"done":true}. NO other action is valid.
7. NEVER repeat the last action in History.
JSON format: {"action_type":"...", "config":{}, "done":false}"""


def parse_action(text: str) -> Action:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return Action(**json.loads(match.group(0)))
    except Exception:
        return Action(action_type=ActionType.INSPECT_LOGS, config={"reason": "parse_error"}, done=False)


def build_prompt(task: str, obs: Observation) -> str:
    """
    Builds a structured, unambiguous prompt from the full observation.
    Every field the LLM needs to make the correct decision is explicit.
    Less ambiguity = fewer wasted steps = fewer API calls.
    """
    last_action = obs.history[-1] if obs.history else "none"
    recent_logs = " | ".join(obs.logs[-3:]) if obs.logs else "No logs"

    # If test_score exists, that's the only thing the LLM should care about
    if obs.test_score is not None:
        return (
            f"Task: {task}\n"
            f"*** TEST SCORE = {obs.test_score}. PIPELINE COMPLETE. ***\n"
            f"Output {{\"action_type\":\"done\",\"config\":{{}},\"done\":true}} and nothing else."
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


def score_is_trustworthy(score, streak, evaluate_confirmed) -> bool:
    return score >= 0.7 and streak >= CONFIRM_STREAK and evaluate_confirmed


def run_task(task: str) -> dict:
    env = MLPipelineEnv(task_name=task)
    client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)
    obs = env.reset()
    rewards: List[float] = []
    steps, success = 0, False
    good_streak = 0
    last_score = None
    evaluate_confirmed = False

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")

    try:
        while steps < MAX_STEPS:

            if "evaluate" in obs.history:
                evaluate_confirmed = True

            # Score stability tracking
            if obs.test_score is not None and obs.test_score >= 0.7:
                if obs.test_score == last_score:
                    good_streak += 1
                else:
                    good_streak = 1
                    print(f"[INFO] test_score={obs.test_score} (streak reset, score changed)")
            else:
                good_streak = 0
            last_score = obs.test_score

            if score_is_trustworthy(obs.test_score or 0.0, good_streak, evaluate_confirmed):
                success = True
                print(f"[INFO] Score {obs.test_score} trusted: "
                      f"streak={good_streak}/{CONFIRM_STREAK}, evaluate_confirmed={evaluate_confirmed}. Done.")
                break

            if obs.test_score is not None and good_streak < CONFIRM_STREAK:
                print(f"[INFO] Confirming score={obs.test_score} "
                      f"(streak={good_streak}/{CONFIRM_STREAK})")

            prompt = build_prompt(task, obs)

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
            obs, reward, done, info = env.step(action)
            rewards.append(reward.total)
            steps += 1

            err = f'"{info.get("error")}"' if info.get("error") else "null"
            print(f"[STEP] step={steps} action={action.model_dump_json()} "
                  f"reward={reward.total:.2f} done={str(done).lower()} error={err}")

            if reward.total <= 0.0 and steps > 3:
                print("[WARN] Reward hit 0.00. Breaking to save credits.")
                break

            if done:
                success = obs.test_score is not None and obs.test_score >= 0.7
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
        print(f"[END] success={str(success).lower()} steps={steps} "
              f"score={score:.2f} rewards={rewards_str}")
        env.close()

    return {"task": task, "success": success, "score": score}


if __name__ == "__main__":
    results = [run_task(t) for t in ["easy", "medium", "hard"]]
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n# Baseline: avg={avg:.2f}", file=sys.stderr)