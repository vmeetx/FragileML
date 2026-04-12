import os
import sys
import json
from typing import List
from openai import OpenAI
from src.environment import MLPipelineEnv
from src.models import Action, ActionType

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = os.getenv("BENCHMARK", "ml-pipeline-env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))

SYSTEM_PROMPT = """You debug ML pipelines. Fix version issues, detect silent failures, prevent data leakage.
Actions: load_model, fix_dependency, preprocess_data, split_data, train_model, evaluate, inspect_logs, validate_data, done.
Respond with JSON only: {"action_type": "fix_dependency", "config": {"package": "transformers"}, "done": false}
Rules: Check logs before acting. Verify with test score. Set done=true when pipeline generalizes."""

def parse_action(text: str) -> Action:
    try:
        start, end = text.find('{'), text.rfind('}') + 1
        if start == -1: raise ValueError
        return Action(**json.loads(text[start:end]))
    except:
        return Action(action_type=ActionType.INSPECT_LOGS, config={}, done=False)

def run_task(task: str) -> dict:
    env = MLPipelineEnv(task_name=task)
    client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL) #check done
    obs = env.reset()
    rewards: List[float] = []
    steps, success = 0, False
    
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")
    try:
        while steps < MAX_STEPS:
            prompt = f"Logs: {obs.logs}\nVal: {obs.validation_score}, Test: {obs.test_score}\nHint: {obs.hint}\nChoose action (JSON):"
            resp = client.chat.completions.create(model=MODEL_NAME, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ], temperature=0.2, max_tokens=150)
            
            action = parse_action(resp.choices[0].message.content)
            obs, reward, done, info = env.step(action)
            rewards.append(reward.total)
            steps += 1
            err = f'"{info.get("error")}"' if info.get("error") else "null"
            print(f"[STEP] step={steps} action={action.model_dump_json()} reward={reward.total:.2f} done={str(done).lower()} error={err}")
            if done:
                success = obs.test_score is not None and obs.test_score > 0.7
                break
    except Exception as e:
        print(f"[STEP] step={steps} action={{\"action_type\":\"done\"}} reward=0.00 done=true error=\"{str(e)}\"")
        rewards.append(0.0)
    finally:
        score = rewards[-1] if rewards else 0.0
        rewards_str = ','.join(f'{r:.2f}' for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")
        env.close()
        return {"task": task, "success": success, "score": score}

if __name__ == "__main__":
    results = [run_task(t) for t in ["easy", "medium", "hard"]]
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n# Baseline: avg={avg:.2f}", file=sys.stderr)