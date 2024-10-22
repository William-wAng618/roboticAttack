from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TVF
import os
def denormalize(images,mean,std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images
path = "/Users/taowenwang/PycharmProjects/demo/openvla-main/run/white_attack/2024_9_3-18-9_unconstrained/adv_noise_iter_0.pt"
im = torch.load(path)
de_im = denormalize(im, mean=torch.tensor([0.484375, 0.455078125, 0.40625]),
                       std=torch.tensor([0.228515625, 0.2236328125, 0.224609375]))

pil_img = TVF.to_pil_image(de_im.squeeze(0))
pil_img.save(path.replace("pt","png"))
a=1
