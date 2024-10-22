import torch
import numpy as np
import torch.nn.functional as F
angle = 30
shx = 0.3
shy = 0.3
def rotation_matrix(theta):
    theta = np.deg2rad(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float32)


# def translation_matrix( tx, ty):
#     return np.array([
#         [1, 0, tx],
#         [0, 1, ty],
#         [0, 0, 1]
#     ], dtype=np.float32)

def shear_matrix(shx,shy):
    return np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def combined_transform_matrix(angle=30,shx=0.3,shy=0.3):
    if np.random.rand() < 0.2:
        return torch.tensor(np.eye(3, dtype=np.float32))
    else:
        angle = np.random.uniform(-angle, angle)
        # tx = np.random.uniform(min_tx, max_tx)
        # ty = np.random.uniform(min_ty, max_ty)
        shx = np.random.uniform(-shx, shx)
        shy = np.random.uniform(-shy, shy)

        R = rotation_matrix(angle)
        # T = translation_matrix(tx, ty)
        S = shear_matrix(shx, shy)
        # combined_matrix = np.dot(T, np.dot(S, R))
        combined_matrix = np.dot(S, R)
        return torch.tensor(combined_matrix)


# 应用仿射变换
def apply_affine_transform(image, transform_matrix):
    if image.ndim == 4:
        image = image.squeeze(0)
    # 提取 2x3 矩阵
    affine_matrix = transform_matrix[:2, :].unsqueeze(0)  # [1, 2, 3]

    # 生成网格
    grid = F.affine_grid(affine_matrix, image.unsqueeze(0).size(), align_corners=False)

    # 应用网格
    transformed_image = F.grid_sample(image.unsqueeze(0), grid, align_corners=False)

    return transformed_image

# 使用示例
from PIL import Image
import torchvision
for i in range(50):
    im = Image.open("/Users/taowenwang/PycharmProjects/demo/openvla-main/0dataprocess/test.png")
    im = torchvision.transforms.ToTensor()(im).unsqueeze(0)

    transfer_matrix = combined_transform_matrix()
    # im [3,224,224]
    im = apply_affine_transform(im, transfer_matrix)
    im = torchvision.transforms.ToPILImage()(im)
    # im.show()
    im.save(f"./{i}.png")