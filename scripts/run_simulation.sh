#!/bin/bash
python evaluation_tool/eval_queue_single_four_spec.py \
    --exp_path PATH TO/fe28658a-4a27-4ffa-82c4-94d44ffc9d48 \
    --cudaid 0 \
    --trials 50 \
    --max_concurrent_tasks 1 \
    --task libero_10 \     
