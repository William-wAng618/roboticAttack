import torch

# a = torch.tensor(
#     [[-100,-100,-100,-100,-100,-100,-100,-100,1,2,3,-100,-100,-100,-100,-100],
#      [-100,-100,-100,-100,4,5,6,-100,-100,-100,-100,-100,-100,-100,-100,-100],
#      [-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,7,8,9,-100,-100]]
# )
#
# b=torch.tensor([99,98,97])
# newlabels = []
# for i in range(a.shape[0]):
#     temp_label = a[i]
#     temp_label[temp_label!=-100] = b
#     newlabels.append(temp_label.unsqueeze(0))
# new_labels = torch.cat(newlabels,dim=0)
# v=1


import sys
from PIL import Image
sys.path.append("/Users/taowenwang/PycharmProjects/demo/openvla-main/attack/white_patch")
from appply_random_transform import RandomPatchTransform

tras = RandomPatchTransform("cpu")
im = Image.open("/Users/taowenwang/PycharmProjects/demo/openvla-main/0dataprocess/test.png")
patch = torch.randn([3,32,32])
mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
modified_images = tras.apply_random_patch_batch([im], patch, mean=mean, std=std)
c=1