import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from prismatic.vla.action_tokenizer import ActionTokenizer

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

image = Image.open("/home/tw9146/tw/openvla-main/0dataprocess/test.png")
prompt = "In: What action should the robot take to put down blue can?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16) #PrismaticProcessor
action_tokenizer = ActionTokenizer(processor.tokenizer)
action = torch.tensor(np.asarray([0,0,0,0,0,0,0])).to("cuda:0")
# labels = action_tokenizer(action)
# c=1
output = vla(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    pixel_values=inputs['pixel_values'],
    return_dict=True,
    labels=action,
)
print(f"output.loss: {output.loss.item()}")
print(f"output.shape: {output.loss.shape}")
# print(f"output: {output.projector_features}")
# print(f"output.shape: {output.projector_features.shape}")
# print(f"Type of output: {type(output.projector_features)}")