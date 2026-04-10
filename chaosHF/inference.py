import os, sys, json, re
from typing import List
from openai import OpenAI
from src.environment import MLPipelineEnv
from src.models import Action, ActionType

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK    = os.getenv("BENCHMARK", "ml-pipeline-env")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "15"))
CONFIRM_STREAK = 2   # score must hold this many consecutive steps before exit

SYSTEM_PROMPT = """You are an ML Pipeline Debugger. Output ONLY valid JSON.
AVAILABLE ACTIONS: load_model, fix_dependency, install_package, preprocess_data,
split_data, train_model, evaluate, inspect_logs, validate_data,
clear_cache, reload_model, reduce_batch, done.

RULES:
1.  model_status='failed', logs mention version mismatch      → fix_dependency {"transformers":"4.25.0"}
2.  model_status='import_error', logs say ModuleNotFoundError → install_package {"package":"<name from log>"}
3.  model_status='partial', logs say Missing keys             → reload_model {"force_download":true}
4.  model_status='loaded', validation unstable across runs    → clear_cache {}
5.  model_status='loaded', logs say CUDA out of memory        → reduce_batch {"batch_size":8}
6.  model_status='loaded', validation_score is null           → train_model {}
7.  task='medium', validation>=0.9, test_score is null        → preprocess_data {"tokenization":true}
8.  task='hard', validation>=0.9, test_score is null          → split_data {"method":"time_series"}
9.  pipeline valid, test_score is null                        → evaluate {"metric":"test"}
10. test_score is not null → {"action_type":"done","config":{},"done":true}. NO other action valid.
11. NEVER repeat the last action in History.
JSON format: {"action_type":"...", "config":{}, "done":false}"""


def parse_action(text: str) -> Action:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return Action(**json.loads(match.group(0)))
    except Exception:
        return Action(action_type=ActionType.INSPECT_LOGS, config={"reason": "parse_error"}, done=False)


def score_is_trustworthy(score: float, streak: int, evaluate_confirmed: bool) -> bool:
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
                    print(f"[INFO] test_score changed to {obs.test_score}. Streak reset to 1.")
            else:
                good_streak = 0
            last_score = obs.test_score

            if score_is_trustworthy(obs.test_score or 0.0, good_streak, evaluate_confirmed):
                success = True
                print(f"[INFO] Score {obs.test_score} confirmed "
                      f"(streak={good_streak}/{CONFIRM_STREAK}, eval_confirmed={evaluate_confirmed}). Done.")
                break

            if obs.test_score is not None and good_streak < CONFIRM_STREAK:
                print(f"[INFO] test_score={obs.test_score} — confirming "
                      f"(streak={good_streak}/{CONFIRM_STREAK})")

            # Build prompt
            last_action = obs.history[-1] if obs.history else "none"
            recent_logs = " | ".join(obs.logs[-3:]) if obs.logs else "No logs"

            if obs.test_score is not None:
                status_line = (
                    f"*** TEST SCORE = {obs.test_score}. "
                    f"OUTPUT done=true NOW. NO other action is valid. ***"
                )
            else:
                status_line = (
                    f"model={obs.model_status} | "
                    f"Val={obs.validation_score} | Test=null"
                )

            prompt = (
                f"Task: {task} | Hint: {obs.hint}\n"
                f"State: {status_line}\n"
                f"Logs: {recent_logs}\n"
                f"Last action: {last_action} — do NOT repeat it.\n"
                f"Choose NEXT action (JSON ONLY):"
            )

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
    all_tasks = ["easy", "medium", "hard", "import_error", "cache_corrupt", "safetensor_fail", "oom_crash"]
    results = [run_task(t) for t in all_tasks]
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n# Baseline: avg={avg:.2f}", file=sys.stderr)