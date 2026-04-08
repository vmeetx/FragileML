```markdown
---
title: FragileML
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🧠 FragileML: ML Pipeline Debugging Environment

An OpenEnv-compatible environment that simulates real-world Hugging Face ML ecosystem failures for AI agent evaluation.

[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/vmeetx/FragileML)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Overview

FragileML trains AI agents to debug fragile ML pipelines by simulating production-ready failure modes:
- **Version Hell**: Dependency conflicts (`transformers`, `torch`)
- **Silent Failures**: Models that load but produce incorrect outputs
- **Data Leakage**: Misleading validation scores masking poor generalization
- **API Misuse**: Wrong parameters causing subtle degradation

Unlike toy environments, FragileML provides deterministic, reproducible scenarios with dense reward shaping for robust agent training.

---

## 🧪 Tasks

| Task | Difficulty | Objective | Max Steps |
|------|------------|-----------|-----------|
| `easy` | 1 | Fix dependency version mismatch to load model | 8 |
| `medium` | 2 | Detect and fix silent tokenization failure | 12 |
| `hard` | 3 | Identify temporal data leakage and correct split | 15 |

---

## 📦 Action Space

```json
{
  "action_type": "load_model|fix_dependency|preprocess_data|split_data|train_model|evaluate|inspect_logs|validate_data|done",
  "config": {"key": "value"},
  "done": false
}
```

---

## 👁️ Observation Space

```json
{
  "dataset_summary": {"name": "string", "size": 1000, "leakage": false},
  "model_status": "not_loaded|loaded|failed|unstable",
  "logs": ["error or status message"],
  "validation_score": 0.85,
  "test_score": 0.82,
  "step_count": 3,
  "available_actions": ["list", "of", "actions"],
  "hint": "Optional guidance string"
}
```

---

## 🚀 Quick Start

### Local Testing
```bash
pip install pydantic openai
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
python inference.py
```

### Docker
```bash
docker build -t fragileml-env .
docker run --rm \
  -e HF_TOKEN="hf_your_token_here" \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
  fragileml-env
```

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | - | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM inference endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-7B-Instruct` | Model identifier |
| `MAX_STEPS` | No | `15` | Max steps per episode |

---

## 📁 Project Structure

```
.
├── Dockerfile
├── inference.py
├── openenv.yaml
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── environment.py
    ├── models.py
    └── tasks.py
```

---

## 📊 Evaluation

This environment is designed to challenge frontier LLM agents with realistic ML engineering scenarios. Baseline zero-shot performance is intentionally low to leave room for RL-trained agents to demonstrate meaningful improvement. The deterministic graders ensure fair, reproducible scoring across all submissions.

---

## 📄 License

MIT License. See LICENSE for details.
```