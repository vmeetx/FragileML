#!/bin/bash
# Quick env setup - run this before testing locally
# Just sets the vars the inference script expects

read -s -p "Enter your HF token (hf_...): " HF_TOKEN
echo ""

if [[ -z "$HF_TOKEN" || ! "$HF_TOKEN" =~ ^hf_ ]]; then
    echo "Hmm, that doesn't look like a valid HF token. Should start with 'hf_'"
    exit 1
fi

export HF_TOKEN="$HF_TOKEN"
export API_BASE_URL="${API_BASE_URL:-https://router.huggingface.co/v1}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export BENCHMARK="ml-pipeline-env"
export MAX_STEPS=15

echo "✅ All set. Run 'python inference.py' or your docker container next."