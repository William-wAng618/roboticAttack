torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/tw9146/tw/openvla-main/bridge_orig \
  --dataset_name bridge_orig \
  --run_root_dir /home/tw9146/tw/openvla-main/run/test \
  --adapter_tmp_dir /home/tw9146/tw/openvla-main/run/test \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla_test \
  --wandb_entity tw \
  --save_steps 1000