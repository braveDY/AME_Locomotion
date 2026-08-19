#!/bin/bash
python scripts/rsl_rl/play.py \
    --task AME-Go2-Custom-Play-v0 \
    --num_envs 1 \
    --video \
    --video_length 300 \
    --save_attention_weights \
    --vis_attention \
    "$@"
