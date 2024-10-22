import torch
import torchvision.transforms.functional as TVF
import numpy as np
from PIL import Image

input_sizes = [[3, 224, 224], [3, 224, 224]]
tvf_resize_params = [
    {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]},
    {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]}
]
tvf_crop_params = [
    {'output_size': [224, 224]},
    {'output_size': [224, 224]}
]
tvf_normalize_params = [
    {'inplace': False, 'mean': [0.484375, 0.455078125, 0.40625],
     'std': [0.228515625, 0.2236328125, 0.224609375]},
    {'inplace': False, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
]
def im_transform(img):
    imgs_t = []
    for idx in range(len(input_sizes)):
        img_idx = TVF.resize(img, **tvf_resize_params[idx])
        img_idx = TVF.center_crop(img_idx, **tvf_crop_params[idx])
        img_idx_t = TVF.to_tensor(img_idx)
        img_idx_t = TVF.normalize(img_idx_t, **tvf_normalize_params[idx])
        imgs_t.append(img_idx_t)
    # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0
    img_t = torch.vstack(imgs_t)
    return img_t

if __name__ == '__main__':
    img = Image.open('test.png')
    im  = TVF.to_tensor(img)
    img = torch.from_numpy(np.asarray(img))
    img_t = im_transform(img)
    a=1