#!/bin/bash
current_dir=$(pwd)
echo $current_dir
python roboticAttack/VLAAttacker/TMA_wrapper.py \
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
    --wandb_project "RELEASE_check" \
    --wandb_entity "taowen_wang-rit" \
    --innerLoop 50 \
    --dataset "libero_spatial" \
    --targetAction 0
