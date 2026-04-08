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
            model_params=dict(init["model"]),  # ✅ RENAMED
            model_status=init["model"]["status"],
            logs=list(init["logs"]),
            validation_score=init["validation_score"],
            test_score=init["test_score"],
            pipeline_valid=False,
            actions_taken=[],
            episode_done=False,
            leakage_detected=init["dataset"].get("leakage", False),
            overfitting_penalty=0.0
        )
        return self._to_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.state.episode_done:
            return self._to_observation(), Reward(total=0.0), True, {"error": "Episode complete"}
        
        self.state.step_count += 1
        self.state.actions_taken.append(action)
        info = {"error": None}
        
        self._apply_action(action, info)
        reward = grade_pipeline(self.state, self.task_config)
        
        if action.done or self.state.step_count >= self.state.max_steps:
            self.state.episode_done = True
        
        done = self.state.episode_done
        return self._to_observation(), reward, done, info

    def state(self) -> State:
        return self.state

    def close(self):
        pass

    def _apply_action(self, action: Action, info: Dict[str, Any]):
        gt = self.task_config["ground_truth"]
        
        if action.action_type == ActionType.FIX_DEPENDENCY:
            if "transformers" in action.config.get("package", ""):
                self.state.model_params["version"] = gt["required_deps"].get("transformers", "4.25.0")  # ✅ RENAMED
                self.state.model_status = "loaded"
                self.state.logs = [l for l in self.state.logs if "Error:" not in l]
                self.state.logs.append("Dependency fixed, model loaded")
                
        elif action.action_type == ActionType.LOAD_MODEL:
            if self.state.model_params.get("version") == gt.get("required_deps", {}).get("transformers"):  # ✅ RENAMED
                self.state.model_status = "loaded"
                self.state.logs.append("Model loaded successfully")
            else:
                self.state.model_status = "failed"
                self.state.logs.append(f"Load failed: version mismatch")
                info["error"] = "Version mismatch"
                
        elif action.action_type == ActionType.PREPROCESS_DATA:
            if self.task_config["name"] == "medium" and "tokenization" in str(action.config):
                self.state.logs.append("Tokenization aligned")
                self.state.validation_score = gt["expected_val"]
                self.state.test_score = gt["corrected_test"]
                self.state.pipeline_valid = True
                
        elif action.action_type == ActionType.SPLIT_DATA:
            if action.config.get("method") == "time_series" and self.task_config["name"] == "hard":
                self.state.leakage_detected = True
                self.state.dataset_config["leakage"] = False
                self.state.logs.append("Temporal split applied, leakage removed")
                self.state.validation_score = 0.74
                self.state.test_score = gt["expected_test_fixed"]
                self.state.pipeline_valid = True
                
        elif action.action_type == ActionType.TRAIN_MODEL:
            if self.state.model_status == "loaded":
                self.state.logs.append("Training completed")
                if not self.state.validation_score:
                    self.state.validation_score = gt["expected_val"]
                    
        elif action.action_type == ActionType.EVALUATE:
            if action.config.get("metric") == "test":
                if self.state.leakage_detected and not self.state.dataset_config.get("leakage_fixed"):
                    self.state.test_score = gt["expected_test_leaked"]
                elif self.state.test_score is None:
                    self.state.test_score = gt.get("expected_test", 0.7)
                self.state.logs.append(f"Test evaluation complete: {self.state.test_score}")
                
        elif action.action_type == ActionType.INSPECT_LOGS:
            if "leakage" in str(self.state.logs) or self.state.dataset_config.get("leakage"):
                self.state.logs.append("Warning: potential data leakage detected")
                
        elif action.action_type == ActionType.VALIDATE_DATA:
            if self.state.dataset_config.get("leakage"):
                self.state.logs.append("Leakage check: FAILED - temporal overlap found")
            else:
                self.state.logs.append("Leakage check: PASSED")

    def _to_observation(self) -> Observation:
        return Observation(
            dataset_summary=self.state.dataset_config,
            model_status=self.state.model_status,
            logs=self.state.logs[-10:],
            validation_score=self.state.validation_score,
            test_score=self.state.test_score,
            history=[f"{a.action_type}:{a.config}" for a in self.state.actions_taken[-5:]],
            step_count=self.state.step_count,
            available_actions=self._get_available_actions(),
            hint=self.task_config["initial"]["hint"]
        )

    def _get_available_actions(self) -> List[str]:
        base = ["load_model", "fix_dependency", "preprocess_data", "split_data", "train_model", "evaluate", "inspect_logs", "validate_data"]
        if self.state.model_status == "loaded" and self.state.validation_score:
            base.append("done")
        return base