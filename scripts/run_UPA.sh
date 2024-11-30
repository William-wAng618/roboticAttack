#!/bin/bash
current_dir=$(pwd)
echo $current_dir
python roboticAttack/VLAAttacker/UPA_wrapper.py \
    --maskidx 0 \
    --lr 2e-3 \
    --server $current_dir \
    --device 0 \
    --iter 2000 \
    --accumulate 1 \
    --bs 8 \
    --warmup 20 \
    --tags "debug testrun" \
    --filterGripTrainTo1 false \
    --geometry true \
    --patch_size "3,50,50" \
    --wandb_project "false" \
    --wandb_project "false" \
    --wandb_entity "xxx" \
    --innerLoop 50 \
    --dataset "libero_spatial" # "libero_spatial" / "libero_10" / "libero_goal" / "libero_goal" / "bridge_orig"