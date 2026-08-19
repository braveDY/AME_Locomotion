#!/bin/bash
python scripts/rsl_rl/train.py \
    --task AME-Go2-Custom-v0 \
    --max_iterations 15000 \
    --headless \
    "$@"