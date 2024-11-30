import torch
from transformers import AutoConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
import os
import numpy as np
import wandb
import argparse
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    import uuid
    exp_id = str(uuid.uuid4())
    import sys
    sys.path.append(f"{args.server}/VLAAttacker/white_patch")
    from TMA import OpenVLAAttacker
    from openvla_dataloader import get_bridge_dataloader,get_dataloader
    if  "bridge_orig" in args.dataset:
        vla_path = "openvla/openvla-7b"
    elif "libero_spatial" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-spatial"
    elif "libero_object" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-object"
    elif "libero_goal" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-goal"
    elif "libero_10" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-10"
    else:
        assert False, "Invalid dataset"
    set_seed(42)
    target = ''
    for i in args.maskidx:
        target += str(i)
    name = f"{args.dataset}_{vla_path}_GA{args.accumulate}_lr{format(args.lr, '.0e')}_iter{args.iter}_warmup{args.warmup}_filterGripTrainTo1{args.filterGripTrainTo1}_target{target}_inner_loop{args.innerLoop}_geometry{args.geometry}_patch_size{args.patch_size}_seed42-{exp_id}"
    if args.wandb_project != "false":
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,name=name, tags=args.tags)
        wandb.config = {"iteration":args.iter, "learning_rate": args.lr, "attack_target": args.maskidx,"accumulate_steps":args.accumulate}
    print(f"exp_id:{exp_id}")
    path = f"{args.server}/run/white_patch_attack/{exp_id}"

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    quantization_config = None
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    vla = vla.to(device)
    os.makedirs(path, exist_ok=True)
    train_dataloader, val_dataloader = get_dataloader(batch_size=args.bs,server=args.server,dataset=args.dataset,vla_path=vla_path)
    openVLA_Attacker = OpenVLAAttacker(vla, processor, path,optimizer="adamW", resize_patch=args.resize_patch)

    # patch 224x224
    # patch_size=[3,22,22] - 1%
    # patch_size=[3,50,50] - 5%
    # patch_size=[3,70,70] - 10%
    # patch_size=[3,87,87] - 15%
    # patch_size=[3,100,100] - 20%
    openVLA_Attacker.patchattack_unconstrained(train_dataloader, val_dataloader, num_iter=args.iter,
                                               target_action=args.targetAction * np.ones(7), patch_size=args.patch_size, alpha=args.lr,
                                               accumulate_steps=args.accumulate,
                                               maskidx=args.maskidx,
                                               warmup=args.warmup,
                                               filterGripTrainTo1=args.filterGripTrainTo1,
                                               geometry=args.geometry,
                                               innerLoop=args.innerLoop,
                                               args=args)

    print("Attack done!")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maskidx',default='0', type=list_of_ints)
    parser.add_argument('--lr',default=2e-3, type=float)
    parser.add_argument('--server',default="xxx", type=str,help='Prefix of the server path')
    parser.add_argument('--device',default=0, type=int)
    parser.add_argument('--iter',default=2000, type=int)
    parser.add_argument('--accumulate',default=1, type=int)
    parser.add_argument('--bs',default=8, type=int)
    parser.add_argument('--warmup',default=20, type=int)
    parser.add_argument('--tags',nargs='+', default=["innerloop50","DoF6","50x50","a6000"])
    parser.add_argument('--filterGripTrainTo1', type=str2bool, nargs='?',default=False,
                        help='Remove the gripper 0 traning samples during the attack of target at grip to 0')
    parser.add_argument('--geometry', type=str2bool, nargs='?',default=True,
                        help='add geometry trans to path')
    parser.add_argument('--patch_size', default='3,50,50', type=list_of_ints)
    parser.add_argument('--wandb_project', default="xxx", type=str)
    parser.add_argument('--wandb_entity', default="xxx", type=str)
    parser.add_argument('--innerLoop', default=50, type=int)
    parser.add_argument('--dataset', default="bridge_orig", type=str)
    parser.add_argument('--resize_patch', type=str2bool, default=False)
    parser.add_argument('--targetAction', default=0, type=float)
    return parser.parse_args()

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def str2bool(value):
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