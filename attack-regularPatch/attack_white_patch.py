import torch
from transformers import AutoConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from datetime import datetime
import os
from PIL import Image
import numpy as np
import wandb
wandb.init(project="openvla_white_regularPatch")

import sys
sys.path.append("/spl_data/tw9146/openvla-main/attack-regularPatch/white_patch")
# from openvla_attacker import OpenVLAAttacker
from openvla_attacker_single import OpenVLAAttacker
from openvla_dataloader import get_bridge_dataloader
import argparse
import random

def set_seed(seed: int):
    # Python内置的随机库
    random.seed(seed)

    # NumPy随机数生成器
    np.random.seed(seed)

    # PyTorch CPU随机数生成器
    torch.manual_seed(seed)

    # 如果有GPU，PyTorch的CUDA随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多卡情况

    # 确保PyTorch中的某些操作是确定性的（可能会导致速度变慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(args):
    # 调用函数设置随机种子
    set_seed(42)
    current_time = datetime.now()
    year = current_time.year
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    path = f"/spl_data/tw9146/openvla-main/run/white_patch_attack/{args.name}_{year}_{month}_{day}-{hour}-{minute}"

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    quantization_config = None
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla = vla.to("cuda:3")
    os.makedirs(path, exist_ok=True)
    train_dataloader, val_dataloader = get_bridge_dataloader(batch_size=8)
    openVLA_Attacker = OpenVLAAttacker(vla, processor, path,optimizer="adamW")

    # patch 224x224
    # patch_size=[3,22,22] - 1%
    # patch_size=[3,50,50] - 5%
    # patch_size=[3,70,70] - 10%
    # patch_size=[3,87,87] - 15%
    # patch_size=[3,100,100] - 20%
    openVLA_Attacker.patchattack_unconstrained(train_dataloader,val_dataloader,num_iter=100000,target_action=np.zeros(7),patch_size=[3,16,16],alpha=1e-3,accumulate_steps=1)
    # openVLA_Attacker.patchattack_unconstrained(train_dataloader,val_dataloader,num_iter=100000,target_action=np.asarray([0,0,0,0,0,0,1]),patch_size=[3,50,50],alpha=1/255,accumulate_steps=64)
    print("Attack done!")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="number of iterations for patch attack")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)