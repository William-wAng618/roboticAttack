import torch
from sympy import expand_power_base
from transformers import AutoConfig
from datetime import datetime
import os
from PIL import Image
import numpy as np
import wandb
import argparse
import random
import sys
sys.path.append("../white_patch/")
from openvla_attacker_single import OpenVLAAttacker
from openvla_dataloader import get_bridge_dataloader
from appply_random_transform import RandomPatchTransform
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
sys.path.append("../../prismatic")
import torchvision
sys.path.append("./massive-activations")
import monkey_patch as mp
import lib
from patch2index import get_patch_emb_index

mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]

# load patch
patch_path = "/spl_data/tw9146/openvla-main/run/white_patch_attack/73cc0c44-cd52-4850-803f-dd3eec888b35/500/patch.pt"
patch = torch.load(patch_path)

# load model
AutoConfig.register("openvla", OpenVLAConfig)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
quantization_config = None
# processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
processor = PrismaticProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    # trust_remote_code=True,
    trust_remote_code=False,
).to("cuda:1")

# load dataset
train_dataloader, val_dataloader = get_bridge_dataloader(batch_size=1, server="/spl_data/tw9146")
data = next(iter(val_dataloader))
randomPatchTransform = RandomPatchTransform(vla.device,resize_patch=False)
modified_images,canvas_list = randomPatchTransform.paste_patch_fix(data["pixel_values"], patch, mean=mean,
                                                                                std=std,inference=True)
modified_images = randomPatchTransform.denormalize(modified_images[:, 0:3, :, :].detach().cpu(), mean=mean[0],
                                                        std=std[0])

# register monkey hook
llama_layers = vla.language_model.model.layers
llama_layer_id = 31 # last 31
mp.enable_llama_custom_decoderlayer(llama_layers[llama_layer_id],llama_layer_id)
dino_layers = vla.vision_backbone.featurizer.blocks
dino_layers_id = 23 # last 23
mp.enable_vit_custom_block(dino_layers, dino_layers_id)
siglip_layers = vla.vision_backbone.fused_featurizer.blocks
siglip_layers_id = 26 # last 26
mp.enable_vit_custom_block(siglip_layers, siglip_layers_id)

os.makedirs(f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_LLAMA/{llama_layer_id+1}",exist_ok=True)
os.makedirs(f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_DINO/{dino_layers_id+1}",exist_ok=True)
os.makedirs(f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_SIG/{siglip_layers_id+1}",exist_ok=True)

stats = {}
# inference
for i in range(modified_images.shape[0]):
    img = torchvision.transforms.ToPILImage()(modified_images[i])
    inputs = processor("What action should the robot take to "+data['instructions'][i], img).to(vla.device, dtype=torch.bfloat16) #PrismaticProcessor
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(f"attacked:{action}")
    seq = "What action should the robot take to " + data['instructions'][i]
    valenc = processor.tokenizer(seq, return_tensors="pt", add_special_tokens=False).input_ids.to(vla.device)
    seq_decoded = ["<s>"]
    for j in range(256):
        seq_decoded.append("<im>")
    for j in range(valenc.shape[1]):
        seq_decoded.append(processor.tokenizer.decode(valenc[0, j].item()))
    for j in range(7):
        seq_decoded.append(f"<DoF{j+1}>")
    stats[f"seq"] = seq_decoded
    llama_feat_abs = torch.cat(llama_layers[llama_layer_id].feat,dim=1).abs()
    stats[f"{llama_layer_id}"] = llama_feat_abs
    dino_feat_abs = dino_layers[dino_layers_id].feat.abs()
    siglip_feat_abs = siglip_layers[siglip_layers_id].feat.abs()
    patch_index = get_patch_emb_index(canvas_list[i])
    print(f"patch_index:{patch_index}")
    lib.plot_3d_feat(obj=stats, layer_id=llama_layer_id, model_name="OpenVLA_Llama", suffix=i, savedir=f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_LLAMA/{llama_layer_id+1}",patch_index=patch_index)
    lib.plot_3d_feat_vit(dino_feat_abs, dino_layers_id, "OpenVLA_DINO", "224", suffix=i,savedir=f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_DINO/{dino_layers_id+1}",patch_index=patch_index)
    lib.plot_3d_feat_vit(siglip_feat_abs, siglip_layers_id, "OpenVLA_SIG", "224", suffix=i,savedir=f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_SIG/{siglip_layers_id+1}",patch_index=patch_index)

    llama_layers[llama_layer_id].feat=[]
    dino_layers[dino_layers_id].feat=[]
    siglip_layers[siglip_layers_id].feat=[]

    # clean
    inputs = processor("What action should the robot take to " + data['instructions'][i], data["pixel_values"][i]).to(vla.device,dtype=torch.bfloat16)  # PrismaticProcessor
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    llama_feat_abs = torch.cat(llama_layers[llama_layer_id].feat, dim=1).abs()
    stats[f"{llama_layer_id}"] = llama_feat_abs
    dino_feat_abs = dino_layers[dino_layers_id].feat.abs()
    siglip_feat_abs = siglip_layers[siglip_layers_id].feat.abs()
    lib.plot_3d_feat(obj=stats, layer_id=llama_layer_id, model_name="OpenVLA_Llama_clean",suffix=i,savedir=f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_LLAMA/{llama_layer_id+1}")
    lib.plot_3d_feat_vit(dino_feat_abs, dino_layers_id, "OpenVLA_DINO_clean", "224", suffix=i,savedir=f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_DINO/{dino_layers_id+1}")
    lib.plot_3d_feat_vit(siglip_feat_abs, siglip_layers_id, "OpenVLA_SIG_clean", "224", suffix=i,savedir=f"/spl_data/tw9146/openvla-main/attack-oneDOF-PositionORrotation/test/save_dir/OpenVLA_SIG/{siglip_layers_id+1}")

    print(f"org:{action}")
    llama_layers[llama_layer_id].feat = []
    dino_layers[dino_layers_id].feat = []
    siglip_layers[siglip_layers_id].feat = []

