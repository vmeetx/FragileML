# ML Pipeline Debugging OpenEnv

Simulates real-world ML engineering failures: version conflicts, silent model issues, data leakage. Agents learn to build robust, generalizing pipelines.

## 🧪 Tasks
| Task | Goal | Key Challenge | Max Steps |
|------|------|--------------|-----------|
| `easy` | Load model by fixing dependency version | Version mismatch detection | 8 |
| `medium` | Fix silent tokenization failure | High val score but wrong outputs | 12 |
| `hard` | Detect & fix temporal data leakage | Misleading validation metric | 15 |

## 📦 Action Space
```json
{
  "action_type": "load_model|fix_dependency|preprocess_data|split_data|train_model|evaluate|inspect_logs|validate_data|done",
  "config": {"key": "value"},
  "done": false
}