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
    import uuid
    exp_id = str(uuid.uuid4())
    import sys
    sys.path.append(f"{args.server}/openvla-main/attack-oneDOF-PositionORrotation/white_patch")
    # from openvla_attacker import OpenVLAAttacker
    from openvla_attacker_single import OpenVLAAttacker
    from openvla_dataloader import get_bridge_dataloader
    # 调用函数设置随机种子
    set_seed(42)
    current_time = datetime.now()
    year = current_time.year
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    target = ''
    for i in args.maskidx:
        target += str(i)
    name = f"random_paste_0-174_GA{args.accumulate}_lr{format(args.lr, '.0e')}_iter{args.iter}_warmup{args.warmup}_filterGripTrainTo1{args.filterGripTrainTo1}_target{target}-{exp_id}"
    wandb_run = wandb.init(entity="taowen_wang-rit", project=args.wandb_project,name=name, tags=args.tags)
    print(f"exp_id:{exp_id}")
    path = f"{args.server}/openvla-main/run/white_patch_attack/{exp_id}"

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
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    vla = vla.to(device)
    os.makedirs(path, exist_ok=True)
    train_dataloader, val_dataloader = get_bridge_dataloader(batch_size=args.bs,server=args.server)
    openVLA_Attacker = OpenVLAAttacker(vla, processor, path,optimizer="adamW")

    # patch 224x224
    # patch_size=[3,22,22] - 1%
    # patch_size=[3,50,50] - 5%
    # patch_size=[3,70,70] - 10%
    # patch_size=[3,87,87] - 15%
    # patch_size=[3,100,100] - 20%
    #  2. Capture a dictionary of hyperparameters
    wandb.config = {"iteration":args.iter, "learning_rate": args.lr, "attack_target": args.maskidx,"accumulate_steps":args.accumulate}
    openVLA_Attacker.patchattack_unconstrained(train_dataloader, val_dataloader, num_iter=args.iter,
                                               target_action=np.zeros(7), patch_size=args.patch_size, alpha=args.lr,
                                               accumulate_steps=args.accumulate,
                                               maskidx=args.maskidx,
                                               warmup=args.warmup,
                                               filterGripTrainTo1=args.filterGripTrainTo1,
                                               geometry=args.geometry,
                                               colorjitter=args.colorjitter)
    # openVLA_Attacker.patchattack_unconstrained(train_dataloader,val_dataloader,num_iter=100000,target_action=np.asarray([0,0,0,0,0,0,1]),patch_size=[3,50,50],alpha=1/255,accumulate_steps=64)

    print("Attack done!")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maskidx',default='6', type=list_of_ints)
    parser.add_argument('--lr',default=2e-3, type=float)
    parser.add_argument('--server',default="/spl_data/tw9146", type=str)
    parser.add_argument('--device',default=0, type=int)
    parser.add_argument('--iter',default=100000, type=int)
    parser.add_argument('--accumulate',default=1, type=int)
    parser.add_argument('--bs',default=8, type=int)
    parser.add_argument('--warmup',default=200, type=int)
    parser.add_argument('--tags',nargs='+', default=["debug"])
    parser.add_argument('--filterGripTrainTo1', type=str2bool, nargs='?',default=False,
                        help='Remove the gripper 0 traning samples during the attack of target at grip to 0')
    parser.add_argument('--geometry', type=str2bool, nargs='?',default=False,
                        help='add geometry trans to path')
    parser.add_argument('--colorjitter', type=str2bool, nargs='?',default=False,
                        help='add colorjitter trans to path')
    parser.add_argument('--patch_size', default='3,50,50', type=list_of_ints)
    parser.add_argument('--wandb_project', default="openvla_white_patch_v2", type=str)
    return parser.parse_args()
def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def str2bool(value):
    # 定义一个函数，将字符串转换为布尔值
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
if __name__ == "__main__":
    args = arg_parser()
    print(f"Paramters:\n maskidx:{args.maskidx}\n lr:{args.lr} \n server:{args.server} \n device:{args.device} \ntags:{args.tags}")
    main(args)