export WANDB_API_KEY='2afe2846a028d2bc651361a4fbdf85cfbd9d9eab'
export WANDB_ENTITY='taowen_wang-rit'
#export WANDB_PROJECT='openvla_white_debug'
#export WANDB_NAME=white_patch_50-50_lr1-255_iter-100000_target-freeze-paste-geo-jitter
#export WANDB_NAME=AdamW_low-G-J_5e-4_GA64_patch-20per
#export WANDB_NAME=fix-region-position-10-10-rolling-dataset-1-255
name='regular_paste_GA1_lr1e-3_100000_targetAtALL'
export WANDB_NAME=$name
#export WANDB_NAME=patch_50_lr1-255_iter-100k_freeze-paste-GA-16
wandb login
python /spl_data/tw9146/openvla-main/attack-regularPatch/attack_white_patch.py --name $name
#python /home/tw9146/tw/openvla-main/attack/attack_white_patch.py
#python ./attack/attack_white_patch.py