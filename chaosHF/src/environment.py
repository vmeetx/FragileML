from typing import Tuple, Optional, Dict, List, Any
from .models import Observation, Action, Reward, State, ActionType
from .tasks import TASKS, grade_pipeline

class MLPipelineEnv:
    def __init__(self, task_name: str = "easy"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")
        self.task_config = TASKS[task_name]
        self.state: Optional[State] = None

    def reset(self) -> Observation:
        init = self.task_config["initial"]
        self.state = State(
            task_name=self.task_config["name"],
            step_count=0,
            max_steps=self.task_config["max_steps"],
            dataset_config=dict(init["dataset"]),
            model_params=dict(init["model"]),
            model_status=init["model"]["status"],
            logs=list(init["logs"]),
            validation_score=init["validation_score"],
            test_score=init["test_score"],
            pipeline_valid=False,
            actions_taken=[],
            episode_done=False,
            leakage_detected=init["dataset"].get("leakage", False),
            overfitting_penalty=0.0,
            consecutive_repeats=0,
            last_action_type=None
        )
        return self._to_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.state.episode_done:
            return self._to_observation(), Reward(total=0.0), True, {"error": "Episode complete"}
        
        # ✅ FIX: Increment step count ALWAYS to penalize inefficiency
        self.state.step_count += 1
        
        # 🔁 Repetition Check
        is_repeat = (
            len(self.state.actions_taken) > 0 and
            self.state.actions_taken[-1].action_type == action.action_type
        )
        
        if is_repeat:
            self.state.consecutive_repeats += 1
            err_msg = f"Repeated action '{action.action_type.value}'. Progress requires new steps."
            self.state.logs.append(f"⚠️ Warning: {err_msg}")
            info = {"error": err_msg}
            reward = grade_pipeline(self.state, self.task_config)
            reward.penalty += 0.15
            reward.total = max(0.0, round(reward.total - 0.15, 2))
            return self._to_observation(), reward, False, info
            
        self.state.consecutive_repeats = 0
        self.state.last_action_type = action.action_type.value
        self.state.actions_taken.append(action)
        
        info = {"error": None}
        self._apply_action(action, info)
        reward = grade_pipeline(self.state, self.task_config)
        
        if action.done or action.action_type == ActionType.DONE or self.state.step_count >= self.state.max_steps:
            self.state.episode_done = True
            
        return self._to_observation(), reward, self.state.episode_done, info

    def _apply_action(self, action: Action, info: Dict[str, Any]):
        gt = self.task_config["ground_truth"]
        cfg = action.config

        if action.action_type == ActionType.FIX_DEPENDENCY:
            # ✅ FIX: Accept 'transformers' key directly in config
            if "transformers" in cfg or cfg.get("package", "").lower() == "transformers":
                self.state.model_params["version"] = gt["required_deps"].get("transformers", "4.25.0")
                self.state.model_status = "loaded"
                self.state.logs = [l for l in self.state.logs if "Error:" not in l]
                self.state.logs.append("✅ Dependency fixed. Next: train_model.")
            else:
                info["error"] = "Invalid dependency config"

        elif action.action_type == ActionType.LOAD_MODEL:
            req_ver = gt.get("required_deps", {}).get("transformers")
            if req_ver and self.state.model_params.get("version") == req_ver:
                self.state.model_status = "loaded"
                self.state.logs.append("✅ Model loaded successfully.")
            else:
                self.state.model_status = "failed"
                self.state.logs.append("❌ Load failed: version mismatch")
                info["error"] = "Version mismatch"

        elif action.action_type == ActionType.PREPROCESS_DATA:
            # ✅ FIX: Check task_name == "medium" (was failing due to trailing space)
            if self.task_config["name"] == "medium" and "tokenization" in str(action.config):
                self.state.logs.append("✅ Tokenization aligned. Pipeline valid.")
                self.state.pipeline_valid = True
            else:
                self.state.logs.append("❌ Invalid preprocess config for this task.")
                info["error"] = "Invalid preprocess config"

        elif action.action_type == ActionType.SPLIT_DATA:
            # ✅ FIX: Check task_name == "hard"
            if self.task_config["name"] == "hard" and action.config.get("method") == "time_series":
                self.state.leakage_detected = False # Leakage fixed
                self.state.dataset_config["leakage_fixed"] = True
                self.state.logs.append("✅ Temporal split applied. Leakage removed.")
                self.state.pipeline_valid = True
            else:
                self.state.logs.append("❌ Invalid split config for this task.")
                info["error"] = "Invalid split config"

        elif action.action_type == ActionType.TRAIN_MODEL:
            if self.state.model_status == "loaded":
                self.state.logs.append("✅ Training completed.")
                if not self.state.validation_score:
                    self.state.validation_score = gt["expected_val"]
                # ✅ FIX: Set valid for Easy task so evaluation can proceed
                if self.task_config["name"] == "easy":
                    self.state.pipeline_valid = True
            else:
                self.state.logs.append("❌ Cannot train: model not loaded")
                info["error"] = "Model not loaded"

        elif action.action_type == ActionType.EVALUATE:
            if cfg.get("metric") == "test":
                if not self.state.pipeline_valid:
                    self.state.test_score = 0.50
                    self.state.logs.append("❌ Evaluation FAILED: Pipeline not fixed. Run fixes first.")
                    info["error"] = "Premature evaluation"
                else:
                    # ✅ FIX: Set correct test score based on task status
                    if self.task_config["name"] == "medium":
                        self.state.test_score = gt.get("corrected_test", 0.79)
                    elif self.task_config["name"] == "hard":
                        self.state.test_score = gt.get("expected_test_fixed", 0.72)
                    else:
                        self.state.test_score = gt.get("expected_test", 0.82)
                    self.state.logs.append(f"✅ Evaluation Success: {self.state.test_score}")
            else:
                self.state.logs.append("❌ Evaluate requires config={'metric': 'test'}")
                info["error"] = "Missing metric config"

        elif action.action_type == ActionType.INSPECT_LOGS:
            pass # No state change, just logs
            self.state.logs.append("🔍 Logs inspected.")

    def _to_observation(self) -> Observation:
        return Observation(
            dataset_summary=self.state.dataset_config,
            model_status=self.state.model_status,
            logs=self.state.logs[-5:],
            validation_score=self.state.validation_score,
            test_score=self.state.test_score,
            history=[a.action_type.value for a in self.state.actions_taken[-5:]],
            step_count=self.state.step_count,
            available_actions=[a.value for a in ActionType],
            hint=self.task_config["initial"]["hint"],
            pipeline_valid=self.state.pipeline_valid,
            leakage_detected=self.state.leakage_detected,
            steps_remaining=self.state.max_steps - self.state.step_count,
        )

    def close(self):
        pass