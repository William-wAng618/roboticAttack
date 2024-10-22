#!/bin/bash
#conda init
#conda activate openvla
nohup python attack/white_patch/check_dataloader.py > attack/white_patch/log/check_dataloader.log
nohup python attack/white_patch/check_dataloader_val.py > attack/white_patch/log/check_dataloader_val.log
