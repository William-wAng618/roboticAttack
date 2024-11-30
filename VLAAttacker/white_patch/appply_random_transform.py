import random
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

class RandomPatchTransform:
    def __init__(self, device, resize_patch):
        self.device = device
        self.angle = 30
        self.shx = 0.2
        self.shy = 0.2
        self.resize_patch = resize_patch

    def normalize(self, images, mean, std):
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def denormalize(self,images, mean, std):
        images = images * std[None, :, None, None]
        images = images + mean[None, :, None, None]
        return images

    def rotation_matrix(self,theta):
        theta = np.deg2rad(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def shear_matrix(self,shx, shy):
        return np.array([
            [1, shx, 0],
            [shy, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def combined_transform_matrix(self):
        if np.random.rand() < 0.2:
            return torch.tensor(np.eye(3, dtype=np.float32))
        else:
            angle = np.random.uniform(-self.angle, self.angle)
            shx = np.random.uniform(-self.shx, self.shx)
            shy = np.random.uniform(-self.shy, self.shy)

            R = self.rotation_matrix(angle)
            S = self.shear_matrix(shx, shy)
            combined_matrix = np.dot(S, R)
            return torch.tensor(combined_matrix)

    def apply_affine_transform(self,image, transform_matrix):
        if image.ndim == 4:
            image = image.squeeze(0)
        affine_matrix = transform_matrix[:2, :].unsqueeze(0)  # [1, 2, 3]

        grid = F.affine_grid(affine_matrix, image.unsqueeze(0).size(), align_corners=False)

        transformed_image = F.grid_sample(image.unsqueeze(0), grid, align_corners=False,padding_mode='border')

        return transformed_image

    def apply_random_patch_batch(self, images, patch, mean, std,geometry):
        modified_images = []
        # apply patch to each image in the batch
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device) * -100

            if self.resize_patch:
                scale = random.uniform(0.61, 1.39) #~ 1%~5%
                height, width = int(patch_height * scale), int(patch_width * scale)  # random scale patch
                patch = transforms.Resize((height, width))(patch)

            patch_channels, patch_height, patch_width = patch.shape

            max_x = img_width - patch_width
            max_y = img_height - patch_height

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            canvas[:, y:y + patch_height, x:x + patch_width] = patch

            if geometry:
                affline_matrix = self.combined_transform_matrix().to(self.device)
                canvas = self.apply_affine_transform(canvas, affline_matrix)

            im = torch.where(canvas < -20, im, canvas)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))

            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)

    def random_paste_patch(self, images, patch, mean, std):
        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device)*-100
            patch_channels, patch_height, patch_width = patch.shape

            max_x = img_width - patch_width
            max_y = img_height - patch_height
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            canvas[:, y:y + patch_height, x:x + patch_width] = patch

            im = torch.where(canvas != -100, canvas, im)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))

            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)

    def paste_patch_fix(self, images, patch, mean, std, inference=False):

        canvas_list = []
        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device)*-100
            patch_channels, patch_height, patch_width = patch.shape

            max_x = img_width - patch_width
            max_y = img_height - patch_height

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            canvas[:, y:y + patch_height, x:x + patch_width] = patch

            im = torch.where(canvas != -100, canvas, im)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))

            modified_images.append(torch.cat([im0,im1],dim=1))
            canvas_list.append(canvas)
        if inference:
            return torch.cat(modified_images, dim=0), canvas_list
        else:
            return torch.cat(modified_images, dim=0)

    def im_process(self, images, mean, std):
        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)