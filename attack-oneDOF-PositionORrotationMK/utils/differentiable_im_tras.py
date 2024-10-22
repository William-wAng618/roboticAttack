# parser.add_argument('--rnd_trans', type=float, default=.1)
# parser.add_argument('--rnd_bri', type=float, default=.3)
# parser.add_argument('--rnd_noise', type=float, default=.02)
# parser.add_argument('--rnd_sat', type=float, default=1.0)
# parser.add_argument('--rnd_hue', type=float, default=.1)
# parser.add_argument('--contrast_low', type=float, default=.5)
# parser.add_argument('--contrast_high', type=float, default=1.5)
# parser.add_argument('--jpeg_quality', type=float, default=25)
# parser.add_argument('--no_jpeg', action='store_true')
# parser.add_argument('--rnd_trans_ramp', type=int, default=10000)
# parser.add_argument('--rnd_bri_ramp', type=int, default=1000)
# parser.add_argument('--rnd_sat_ramp', type=int, default=1000)
# parser.add_argument('--rnd_hue_ramp', type=int, default=1000)
# parser.add_argument('--rnd_noise_ramp', type=int, default=1000)
# parser.add_argument('--contrast_ramp', type=int, default=1000)

import torch
import torchvision.transforms.functional as TVF
input_sizes = [[3,224,224],[3,224,224]]
tvf_resize_params = [
    {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]},
    {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]}
]
tvf_crop_params = [
    {'output_size': [224, 224]},
    {'output_size': [224, 224]}
]
tvf_normalize_params = [
    {'inplace': False, 'mean': [0.484375, 0.455078125, 0.40625], 'std': [0.228515625, 0.2236328125, 0.224609375]},
    {'inplace': False, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
]
for idx in range(len(input_sizes)):
    img_idx = TVF.resize(img, **tvf_resize_params[idx])
    img_idx = TVF.center_crop(img_idx, **tvf_crop_params[idx])
    img_idx_t = TVF.to_tensor(img_idx)
    img_idx_t = TVF.normalize(img_idx_t, **tvf_normalize_params[idx])
    imgs_t.append(img_idx_t)