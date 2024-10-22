#!/bin/bash
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --exp_name a5083c2b-1186-4464-ab9f-1056211a2221 \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --run_id_note a5083c2b-1186-4464-ab9f-1056211a2221 \
    --use_wandb True \
    --wandb_project LIBERO_simulation_test \
    --wandb_entity taowen_wang-rit