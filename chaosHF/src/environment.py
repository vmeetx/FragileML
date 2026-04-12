# src/environment.py 
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

        self.state.step_count += 1

        # 🔥 HARD FIX: prevent evaluate before train_model (auto-correct)
        if action.action_type == ActionType.EVALUATE:
            if self.task_config["name"] in ["easy", "hard"]:
                prior_actions = [a.action_type for a in self.state.actions_taken]
                if ActionType.TRAIN_MODEL not in prior_actions:
                    action = Action(
                        action_type=ActionType.TRAIN_MODEL,
                        config={},
                        done=False
                    )

        # Repetition check with strong penalty
        is_repeat = (
            len(self.state.actions_taken) > 0
            and self.state.actions_taken[-1].action_type == action.action_type
        )
        if is_repeat:
            self.state.consecutive_repeats += 1
            if self.state.consecutive_repeats >= 2:
                err_msg = f"Action '{action.action_type.value}' repeated — progress requires new steps"
                self.state.logs.append(f"⚠️ {err_msg}")
                reward = Reward(total=0.0, pipeline_score=0.0, generalization_score=0.0, efficiency_score=0.0, penalty=0.50, info="Repeated action blocked")
                return self._to_observation(), reward, False, {"error": err_msg}
        else:
            self.state.consecutive_repeats = 0
        
        self.state.last_action_type = action.action_type.value
        self.state.actions_taken.append(action)
        info = {"error": None}
        
        if not self._validate_action_preconditions(action, info):
            if action.action_type == ActionType.EVALUATE and info.get("error"):
                self.state.test_score = 0.40
            reward = Reward(total=0.0, pipeline_score=0.0, generalization_score=0.0, efficiency_score=0.0, penalty=0.40, info=info["error"])
            self.state.logs.append(f"❌ {info['error']}")
            return self._to_observation(), reward, False, info

        self._apply_action(action, info)
        reward = grade_pipeline(self.state, self.task_config, action.action_type.value)

        if action.done or action.action_type == ActionType.DONE:
            if not (self.state.pipeline_valid and self.state.test_score is not None):
                info["error"] = "Cannot end: pipeline not fully validated"
                self.state.logs.append("⚠️ Episode cannot end — pipeline incomplete")
                reward = Reward(
                    total=max(0.0, reward.total - 0.30),
                    pipeline_score=reward.pipeline_score,
                    generalization_score=reward.generalization_score,
                    efficiency_score=reward.efficiency_score,
                    penalty=reward.penalty + 0.30,
                    info="Premature termination blocked"
                )
                return self._to_observation(), reward, False, info
            self.state.episode_done = True
        elif self.state.step_count >= self.state.max_steps:
            self.state.episode_done = True

        return self._to_observation(), reward, self.state.episode_done, info

    def _validate_action_preconditions(self, action: Action, info: Dict[str, Any]) -> bool:
        gt = self.task_config["ground_truth"]
        if action.action_type == ActionType.FIX_DEPENDENCY:
            if "transformers" not in str(action.config) and action.config.get("package") != "transformers":
                info["error"] = "fix_dependency requires transformers package configuration"
                return False
            return True
        elif action.action_type == ActionType.LOAD_MODEL:
            req_ver = gt.get("required_deps", {}).get("transformers")
            if req_ver and self.state.model_params.get("version") != req_ver:
                info["error"] = f"Cannot load: transformers version must be {req_ver}"
                return False
            return True
        elif action.action_type == ActionType.TRAIN_MODEL:
            if self.state.model_status != "loaded":
                info["error"] = "Cannot train: model not loaded (resolve dependencies first)"
                return False
            return True
        elif action.action_type == ActionType.PREPROCESS_DATA:
            if self.task_config["name"] == "medium":
                if "tokenization" not in str(action.config):
                    info["error"] = "preprocess_data requires tokenization fix for this task"
                    return False
            return True
        elif action.action_type == ActionType.SPLIT_DATA:
            if self.task_config["name"] == "hard":
                if action.config.get("method") != "time_series":
                    info["error"] = "split_data requires method='time_series' to address leakage"
                    return False
            return True
        elif action.action_type == ActionType.EVALUATE:
            if action.config.get("metric") != "test":
                info["error"] = "evaluate requires config={'metric': 'test'}"
                return False
            if not self.state.pipeline_valid:
                info["error"] = "Cannot evaluate: pipeline not valid (complete required steps first)"
                return False
            if self.task_config["name"] in ["easy", "hard"]:
                prior_actions = [a.action_type for a in self.state.actions_taken]
                if ActionType.TRAIN_MODEL not in prior_actions:
                    info["error"] = "Cannot evaluate: train_model required before evaluation"
                    return False
            return True
        elif action.action_type == ActionType.DONE:
            if not (self.state.pipeline_valid and self.state.test_score is not None):
                info["error"] = "Cannot end: pipeline not fully validated"
                return False
            return True
        return True

    def _apply_action(self, action: Action, info: Dict[str, Any]):
        gt = self.task_config["ground_truth"]
        cfg = action.config
        
        if action.action_type == ActionType.FIX_DEPENDENCY:
            self.state.model_params["version"] = gt["required_deps"].get("transformers", "4.25.0")
            self.state.model_status = "loaded"
            self.state.logs = [l for l in self.state.logs if "Error:" not in l and "failed" not in l.lower()]
            self.state.logs.append("✅ Dependency issue resolved — model should now load correctly")
        elif action.action_type == ActionType.LOAD_MODEL:
            self.state.model_status = "loaded"
            self.state.logs.append("✅ Model loaded successfully — weights initialized")
        elif action.action_type == ActionType.TRAIN_MODEL:
            self.state.logs.append("✅ Training completed — validation metrics updated")
            if self.state.validation_score is None:
                self.state.validation_score = gt["expected_val"]
            if self.task_config["name"] == "easy":
                self.state.pipeline_valid = True
        elif action.action_type == ActionType.PREPROCESS_DATA:
            if self.task_config["name"] == "medium" and "tokenization" in str(cfg):
                self.state.logs.append("✅ Data preprocessing applied — tokenization aligned")
                self.state.pipeline_valid = True
            else:
                self.state.logs.append("⚠️ Preprocessing applied — verify tokenization alignment")
        elif action.action_type == ActionType.SPLIT_DATA:
            if self.task_config["name"] == "hard" and cfg.get("method") == "time_series":
                self.state.leakage_detected = False
                self.state.dataset_config["leakage_fixed"] = True
                self.state.logs.append("✅ Data split applied — temporal ordering preserved")
                self.state.pipeline_valid = True
            else:
                self.state.logs.append("⚠️ Split applied — verify leakage mitigation")
        elif action.action_type == ActionType.EVALUATE:
            if self.task_config["name"] == "medium":
                base_score = gt["expected_test_fixed"] if self.state.pipeline_valid else gt["expected_test_broken"]
            elif self.task_config["name"] == "hard":
                base_score = gt["expected_test_fixed"] if (self.state.pipeline_valid and not self.state.leakage_detected) else gt["expected_test_leaked"]
            else:
                base_score = gt["expected_test"]
            
            self.state.test_score = base_score
            
            if self.state.pipeline_valid:
                self.state.logs.append(f"✅ Evaluation complete — test_score={self.state.test_score}")
            else:
                self.state.logs.append(f"⚠️ Evaluation completed — note: metrics may reflect unresolved issues (score={self.state.test_score})")
        elif action.action_type == ActionType.INSPECT_LOGS:
            self.state.logs.append("🔍 Log review complete — error patterns identified")

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

    def state(self) -> State:
        return self.state

    def close(self):
        pass