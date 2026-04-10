from typing import Tuple, Optional, Dict, List, Any
from .models import Observation, Action, Reward, State, ActionType
from .tasks import TASKS, grade_pipeline


class MLPipelineEnv:
    def __init__(self, task_name: str = "easy"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
        self.task_config = TASKS[task_name]
        self.state: Optional[State] = None

    def reset(self) -> Observation:
        init = self.task_config["initial"]
        gt   = self.task_config["ground_truth"]
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
            last_action_type=None,
            # extended fields
            cache_cleared=False,
            weights_valid=self.task_config["name"] != "safetensor_fail",
            batch_size=gt.get("oom_batch_size"),
            oom_triggered=False,
        )
        return self._to_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.state.episode_done:
            return self._to_observation(), Reward(total=0.0), True, {"error": "Episode complete"}

        self.state.step_count += 1

        # Repetition check
        is_repeat = (
            len(self.state.actions_taken) > 0 and
            self.state.actions_taken[-1].action_type == action.action_type
        )
        if is_repeat:
            self.state.consecutive_repeats += 1
            err_msg = f"Repeated action '{action.action_type.value}'. Progress requires new steps."
            self.state.logs.append(f"⚠️ Warning: {err_msg}")
            reward = grade_pipeline(self.state, self.task_config)
            reward.penalty += 0.15
            reward.total = max(0.0, round(reward.total - 0.15, 2))
            return self._to_observation(), reward, False, {"error": err_msg}

        self.state.consecutive_repeats = 0
        self.state.last_action_type = action.action_type.value
        self.state.actions_taken.append(action)

        info = {"error": None}
        self._apply_action(action, info)
        reward = grade_pipeline(self.state, self.task_config)

        if action.done or action.action_type == ActionType.DONE \
                or self.state.step_count >= self.state.max_steps:
            self.state.episode_done = True

        return self._to_observation(), reward, self.state.episode_done, info

    def _apply_action(self, action: Action, info: Dict[str, Any]):
        gt  = self.task_config["ground_truth"]
        cfg = action.config
        task = self.task_config["name"]

        # ── Shared actions ───────────────────────────────────────────────────

        if action.action_type == ActionType.FIX_DEPENDENCY:
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
            if task == "medium" and "tokenization" in str(cfg):
                self.state.logs.append("✅ Tokenization aligned. Pipeline valid.")
                self.state.pipeline_valid = True
            else:
                self.state.logs.append("❌ Invalid preprocess config for this task.")
                info["error"] = "Invalid preprocess config"

        elif action.action_type == ActionType.SPLIT_DATA:
            if task == "hard" and cfg.get("method") == "time_series":
                self.state.leakage_detected = False
                self.state.dataset_config["leakage_fixed"] = True
                self.state.logs.append("✅ Temporal split applied. Leakage removed.")
                self.state.pipeline_valid = True
            else:
                self.state.logs.append("❌ Invalid split config for this task.")
                info["error"] = "Invalid split config"

        elif action.action_type == ActionType.TRAIN_MODEL:
            if task == "oom_crash":
                # OOM: training only succeeds if batch was already reduced
                if self.state.batch_size is not None and self.state.batch_size <= gt["safe_batch_size"]:
                    self.state.logs.append(f"✅ Training succeeded with batch_size={self.state.batch_size}.")
                    self.state.validation_score = gt["expected_val"]
                    self.state.pipeline_valid = True
                else:
                    self.state.oom_triggered = True
                    self.state.logs.append(
                        f"❌ OOM again. batch_size={self.state.batch_size} too large. Reduce it first."
                    )
                    info["error"] = "OOM: batch size too large"
            elif self.state.model_status == "loaded":
                self.state.logs.append("✅ Training completed.")
                if not self.state.validation_score:
                    self.state.validation_score = gt.get("expected_val", gt.get("expected_val_after_fix"))
                if task == "easy":
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
                    if task == "medium":
                        self.state.test_score = gt.get("corrected_test", 0.79)
                    elif task == "hard":
                        self.state.test_score = gt.get("expected_test_fixed", 0.72)
                    elif task == "cache_corrupt":
                        # Without clearing cache: noisy result; with clear: clean result
                        self.state.test_score = gt["expected_test"] if self.state.cache_cleared else 0.61
                    elif task == "safetensor_fail":
                        # Without valid weights: garbage; with valid weights: good
                        self.state.test_score = gt["expected_test"] if self.state.weights_valid else 0.41
                    else:
                        self.state.test_score = gt.get("expected_test", 0.82)
                    self.state.logs.append(f"✅ Evaluation: test_score={self.state.test_score}")
            else:
                self.state.logs.append("❌ Evaluate requires config={'metric': 'test'}")
                info["error"] = "Missing metric config"

        elif action.action_type == ActionType.INSPECT_LOGS:
            self.state.logs.append("🔍 Logs inspected.")

        # ── New actions for extended tasks ───────────────────────────────────

        elif action.action_type == ActionType.INSTALL_PACKAGE:
            # import_error: agent must name the correct missing package
            pkg = cfg.get("package", "")
            if task == "import_error" and pkg == gt["missing_package"]:
                self.state.model_status = "loaded"
                self.state.logs.append(f"✅ Package '{pkg}' installed. Model loaded successfully.")
                self.state.pipeline_valid = False  # still needs training
            else:
                self.state.logs.append(f"❌ Wrong package '{pkg}'. Read the error logs carefully.")
                info["error"] = f"Wrong package: {pkg}"

        elif action.action_type == ActionType.CLEAR_CACHE:
            # cache_corrupt: clears stale cache and re-loads model cleanly
            if task == "cache_corrupt":
                self.state.cache_cleared = True
                self.state.model_status = "loaded"
                self.state.validation_score = gt["expected_val_after_fix"]
                self.state.logs.append("✅ Cache cleared. Model reloaded. Validation stable: "
                                       f"{gt['expected_val_after_fix']}")
                self.state.pipeline_valid = True
            else:
                self.state.logs.append("❌ clear_cache not applicable to this task.")
                info["error"] = "Action not applicable"

        elif action.action_type == ActionType.RELOAD_MODEL:
            # safetensor_fail: force fresh download to fix missing weight keys
            if task == "safetensor_fail" and cfg.get("force_download") is True:
                self.state.weights_valid = True
                self.state.model_status = "loaded"
                self.state.logs.append("✅ Model reloaded with force_download=True. All weights present.")
                self.state.pipeline_valid = False  # still needs training
            else:
                self.state.logs.append("❌ reload_model requires config={'force_download': true}.")
                info["error"] = "Missing force_download config"

        elif action.action_type == ActionType.REDUCE_BATCH:
            # oom_crash: agent must pick a batch size <= safe_batch_size
            new_batch = cfg.get("batch_size")
            if task == "oom_crash" and isinstance(new_batch, int):
                if new_batch <= gt["safe_batch_size"]:
                    self.state.batch_size = new_batch
                    self.state.logs.append(
                        f"✅ Batch size reduced to {new_batch}. Now run train_model."
                    )
                else:
                    self.state.logs.append(
                        f"⚠️ batch_size={new_batch} still too large "
                        f"(safe <= {gt['safe_batch_size']}). Reduce further."
                    )
                    info["error"] = f"batch_size {new_batch} still too large"
            else:
                self.state.logs.append("❌ reduce_batch requires config={'batch_size': <int>}.")
                info["error"] = "Invalid reduce_batch config"

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
            hint=self.task_config["initial"]["hint"]
        )

    def close(self):
        pass