---
title: FragileML - ML Pipeline Debugging Environment
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: ""
python_version: "3.11"
pinned: false
---

# 🧠 FragileML: ML Pipeline Debugging Environment

<div align="center">

**An OpenEnv-compatible environment that simulates real-world Hugging Face ML ecosystem failures**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.1.0-blue)](https://github.com/huggingface/openenv)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

**[Live Demo](https://huggingface.co/spaces/Vmeetx/FragileMLHackathon)** | **[GitHub](https://github.com/vmeetx/FragileML)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Key Features](#-key-features)
- [Tasks](#-tasks)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Baseline Results](#-baseline-results)
- [Project Structure](#-project-structure)
- [Compliance Checklist](#-compliance-checklist)

---

## 🎯 Overview

**FragileML** is a production-grade OpenEnv environment that trains AI agents to debug and repair fragile ML pipelines. It simulates the exact failure modes that ML engineers face daily when working with Hugging Face ecosystems: version conflicts, silent model failures, data leakage, and API misuse.

Unlike toy environments, FragileML provides **deterministic, reproducible failure scenarios** with **dense reward signals** that enable agents to learn robust ML engineering skills.

---

## 💥 Problem Statement

### Real-World ML Engineering Challenges

ML practitioners constantly battle:

1. **Version Hell** - Dependency conflicts between `transformers`, `torch`, and `accelerate`
2. **Silent Failures** - Models that load successfully but produce incorrect outputs
3. **Data Leakage** - Validation scores that lie, masking poor generalization
4. **API Misuse** - Wrong parameters causing subtle performance degradation
5. **Cache Corruption** - Stale weights producing inconsistent results

### The Gap

Existing RL environments focus on games or synthetic tasks. **No environment exists** that trains agents on real ML engineering workflows with realistic failure modes and proper reward shaping.

---

## 💡 Solution

We built a **deterministic state machine** that simulates the Hugging Face ML ecosystem without requiring heavy ML libraries. The environment:

- ✅ **Models real failure modes** (not random noise)
- ✅ **Provides dense rewards** (partial progress signals)
- ✅ **Ensures reproducibility** (deterministic graders)
- ✅ **Runs on minimal hardware** (<500MB RAM, 2 vCPU)
- ✅ **Follows OpenEnv spec** (typed models, standard API)

---

## 🌟 Key Features

### 🔬 Realistic Failure Simulation

| Failure Mode | Simulation | Real-World Analogy |
|--------------|------------|-------------------|
| **Version Mismatch** | `transformers==4.30.0` fails when `4.25.0` required | `pip install` conflicts |
| **Silent Tokenization Bug** | High validation (0.91) but low test score (0.68) | Misaligned tokenizers |
| **Temporal Data Leakage** | Validation R²=0.98, Test R²=0.45 | Time-series split errors |
| **Cache Corruption** | Model loads but outputs inconsistent | Stale `.cache/huggingface` |
| **API Parameter Misuse** | Missing `max_length` causes truncation | Wrong pipeline config |

### 🎯 Dense Reward Shaping

```python
# Multi-component reward (not sparse binary)
total = (
    pipeline_score        # +0.4 for valid pipeline steps
    + generalization      # +0.3 for test score close to target
    + efficiency          # +0.15 for finishing early
    - overfitting_penalty # -0.15 if val-test gap > 0.3
    - redundancy_penalty  # -0.05 per useless action
)
