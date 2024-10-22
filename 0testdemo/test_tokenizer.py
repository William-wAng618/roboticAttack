import numpy as np
import torch
from transformers import AutoConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
import sys
import time
from tqdm import tqdm
sys.path.append("/spl_data/tw9146/openvla-main/attack-oneDOF/white_patch")
from openvla_dataloader import get_bridge_dataloader
org_target_action = np.array([-1,0,1,0,0,0,0])
AutoConfig.register("openvla", OpenVLAConfig)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
base_tokenizer = processor.tokenizer
action_tokenizer = ActionTokenizer(processor.tokenizer)
target_action = base_tokenizer(action_tokenizer(org_target_action)).input_ids[2:]
# decode_action = action_tokenizer.decode_token_ids_to_actions(np.asarray(target_action))
# decode_action2 = action_tokenizer.decode_token_ids_to_actions(np.asarray([32000, 31872, 31743, 31872, 31872, 31872, 31872]))
# a=1
train_dataloader, val_dataloader = get_bridge_dataloader(batch_size=8)
for batch in tqdm(train_dataloader):
    # if batch==12500:
    #     a=1
    labels = batch['labels']
    for idx in range(labels.size(0)):
        templabel = labels[idx]
        mask = templabel > 2
        templabel = templabel[mask]
        if templabel.equal(torch.tensor([31872,31872,31872,31872,31872,31872,31872])):
            print(f"find! all zero action:{templabel}")
            break