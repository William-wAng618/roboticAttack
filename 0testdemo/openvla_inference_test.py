# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
def denormalize(images,mean,std):
    # mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device)
    # std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# # Grab image input & format prompt
# image: Image.Image = get_from_camera(...)
image = Image.open("/spl_data/tw9146/openvla-main/0dataprocess/test.png")
prompt = "In: What action should the robot take to put down blue can?\nOut:"
#
# # Predict Action (7-DoF; un-normalize for BridgeData V2)
# inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16) #PrismaticProcessor
im_tensor1 = inputs["pixel_values"][:,:3,:,:]
im_tensor2 = inputs["pixel_values"][:,3:,:,:]
de_im_tensor1 = denormalize(im_tensor1,torch.tensor([ 0.484375,  0.455078125,  0.40625]).to(im_tensor1.device),torch.tensor([0.228515625,0.2236328125, 0.224609375]).to(im_tensor1.device))
de_im_tensor2 = denormalize(im_tensor2,torch.tensor([0.5,0.5,0.5]).to(im_tensor2.device),torch.tensor([0.5,0.5,0.5]).to(im_tensor2.device))
print("send into llm")
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(f"Predicted Action: {action}")
print(f"Type: {type(action)}")
print(f"Shape: {action.shape}")
# print(f"Type(VLA):{type(vla)}")

