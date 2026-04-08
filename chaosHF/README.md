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