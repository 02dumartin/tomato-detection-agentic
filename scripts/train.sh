#!/bin/bash
# DETR Training Script - Sequential Execution
# 3-class 먼저 실행 후, 1-class 실행

set -e

# 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "DETR Multi-Config Training"
echo "========================================"
echo ""

# 1. 3-class 학습
echo ">>> Starting 3-class training..."
python scripts/train.py --config configs/detr/detr_3class.yaml
echo ">>> 3-class training completed!"
echo ""

# 2. 1-class 학습
echo ">>> Starting 1-class training..."
python scripts/train.py --config configs/detr/detr_1class.yaml
echo ">>> 1-class training completed!"
echo ""

echo "========================================"
echo "All trainings completed!"
echo "========================================"
