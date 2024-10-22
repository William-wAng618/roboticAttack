import random
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def denormalize(images,mean,std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

class RandomPatchTransform:
    def __init__(self, device):
        self.device = device
        self.angle = 30
        self.shx = 0.2
        self.shy = 0.2

    # Normalize
    def normalize(self, images, mean, std):
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def denormalize(self,images, mean, std):
        images = images * std[None, :, None, None]
        images = images + mean[None, :, None, None]
        return images


    # Geometry TRANSFORMATIONS
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
            # tx = np.random.uniform(min_tx, max_tx)
            # ty = np.random.uniform(min_ty, max_ty)
            shx = np.random.uniform(-self.shx, self.shx)
            shy = np.random.uniform(-self.shy, self.shy)

            R = self.rotation_matrix(angle)
            # T = translation_matrix(tx, ty)
            S = self.shear_matrix(shx, shy)
            # combined_matrix = np.dot(T, np.dot(S, R))
            combined_matrix = np.dot(S, R)
            return torch.tensor(combined_matrix)

    # 应用仿射变换
    def apply_affine_transform(self,image, transform_matrix):
        if image.ndim == 4:
            image = image.squeeze(0)
        # 提取 2x3 矩阵
        affine_matrix = transform_matrix[:2, :].unsqueeze(0)  # [1, 2, 3]

        # 生成网格
        grid = F.affine_grid(affine_matrix, image.unsqueeze(0).size(), align_corners=False)

        # 应用网格
        transformed_image = F.grid_sample(image.unsqueeze(0), grid, align_corners=False,padding_mode='border')

        return transformed_image

    def get_color_jitter_params(self, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1), saturation_range=(0.9, 1.1),
                            hue_range=(-0.1, 0.1)):
        # 随机生成调整因子
        brightness_factor = torch.FloatTensor(1).uniform_(*brightness_range).item()
        contrast_factor = torch.FloatTensor(1).uniform_(*contrast_range).item()
        saturation_factor = torch.FloatTensor(1).uniform_(*saturation_range).item()
        hue_factor = torch.FloatTensor(1).uniform_(*hue_range).item()
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def random_color_jitter(self, image, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        """
        Random color dithering, including brightness, contrast, saturation, and hue adjustments.
        Only applies transformations to non-zero regions of the image.

        Args:
            image (Tensor):  [C, H, W] (image tensor)
            brightness_factor (float): Brightness adjustment factor
            contrast_factor (float): Contrast adjustment factor
            saturation_factor (float): Saturation adjustment factor
            hue_factor (float): Hue adjustment factor

        Returns:
            Tensor: Adjusted image
        """

        # 创建非零像素的掩码
        non_zero_mask = image > -20
        non_zero_pixels = image[non_zero_mask]

        # 如果存在非零像素，进行计算和调整
        if len(non_zero_pixels) > 0:
            # brightness adjustment
            non_zero_pixels = non_zero_pixels * brightness_factor

            # contrast adjustment
            non_zero_mean = non_zero_pixels.mean()
            non_zero_pixels = (non_zero_pixels - non_zero_mean) * contrast_factor + non_zero_mean

            # saturation adjustment
            non_zero_gray = non_zero_pixels.mean(dim=0, keepdim=True)
            non_zero_pixels = (non_zero_pixels - non_zero_gray) * saturation_factor + non_zero_gray

            # hue adjustment
            hue = torch.tensor(hue_factor, dtype=image.dtype)
            non_zero_pixels = (non_zero_pixels + hue) % 1.0
            image[non_zero_mask] = non_zero_pixels

        return image


    def apply_random_patch_batch(self, images, patch, mean, std,geometry,colorjitter):
        """
        random paste patch to images

        param:
        images (torch.Tensor):  list PIL images
        patch (torch.Tensor):  [3, patch_height, patch_width] patch

        return:
        torch.Tensor: batch img with patch added
        """

        modified_images = []
        # apply patch to each image in the batch
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device) * -100
            patch_channels, patch_height, patch_width = patch.shape

            # scale = random.uniform(0.8, 1.2)
            # height, width = int(patch_height * scale), int(patch_width * scale)  # random scale patch
            # patch = transforms.Resize((height, width))(patch)
            #
            # # 补丁限制位置
            # max_x = img_width - int(patch_width * scale)
            # max_y = img_height - int(patch_height * scale)
            #
            # # 随机选择补丁的起始位置
            # x = random.randint(0, max_x)
            # y = random.randint(0, max_y)
            # # 将补丁粘贴到随机位置
            # canvas[:, y:y + int(patch_height * scale), x:x + int(patch_width * scale)] = patch

            # 补丁限制位置
            max_x = img_width - patch_width
            max_y = img_height - patch_height

            # 随机选择补丁的起始位置
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            # 将补丁粘贴到随机位置
            canvas[:, y:y + patch_height, x:x + patch_width] = patch

            if geometry:
                # Geometry transform
                affline_matrix = self.combined_transform_matrix().to(self.device)
                canvas = self.apply_affine_transform(canvas, affline_matrix)

            if colorjitter:
                # Color transform
                color_jitter_params = self.get_color_jitter_params()
                canvas = self.random_color_jitter(canvas, *color_jitter_params)

            im = torch.where(canvas < -20, im, canvas)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))

            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)

    def random_paste_patch(self, images, patch, mean, std):
        """
        random paste patch to images

        param:
        images (torch.Tensor):  list PIL images
        patch (torch.Tensor):  [3, patch_height, patch_width] patch
        return:
        torch.Tensor: batch img with patch added
        """

        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device)*-100
            patch_channels, patch_height, patch_width = patch.shape

            # 随机选择补丁的起始位置
            max_x = img_width - patch_width
            max_y = img_height - patch_height
            # 随机选择补丁的起始位置
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            # 将补丁粘贴到随机位置
            canvas[:, y:y + patch_height, x:x + patch_width] = patch

            im = torch.where(canvas != -100, canvas, im)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))

            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)

    def paste_patch_fix(self, images, patch, mean, std, inference=False):
        """
        random paste patch to images

        param:
        images (torch.Tensor):  list PIL images
        patch (torch.Tensor):  [3, patch_height, patch_width] patch
        return:
        torch.Tensor: batch img with patch added
        """
        canvas_list = []
        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device)*-100
            patch_channels, patch_height, patch_width = patch.shape

            # 随机选择补丁的起始位置

            # center
            # 随机选择补丁的起始位置
            max_x = img_width - patch_width
            max_y = img_height - patch_height

            # max_x = 10
            # max_y = 10
            # 随机选择补丁的起始位置
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            # 将补丁粘贴到随机位置
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
        """
        process images

        param:
        images (torch.Tensor):  list PIL images
        return:
        torch.Tensor: batch processed img
        """
        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)

    def paste_patch_fix2(self, images, patch, mean, std):
        """
        random paste patch to images

        param:
        images (torch.Tensor):  list PIL images
        patch (torch.Tensor):  [3, patch_height, patch_width] patch
        return:
        torch.Tensor: batch img with patch added
        """

        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device)*-100
            patch_channels, patch_height, patch_width = patch.shape

            # 随机选择补丁的起始位置

            # center
            # 随机选择补丁的起始位置
            # max_x = img_width - patch_width
            # max_y = img_height - patch_height

            x = 10
            y = 10

            # 将补丁粘贴到随机位置
            canvas[:, y:y + patch_height, x:x + patch_width] = patch

            im = torch.where(canvas != -100, canvas, im)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))

            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)

    def simulation_paste_patch(self,image,patch,random=False,geometry=False,colorjitter=False):
        """
        random paste patch to images

        param:
        image (numpy.ndarray):  ndarray image [224,224,3]
        patch (torch.Tensor):  [3, patch_height, patch_width] patch
        return: torch.Tensor: batch img with patch added
        """
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        # paste patch
        if random:
            img_channels, img_height, img_width = im.shape
            patch_channels, patch_height, patch_width = patch.shape
            max_x = img_width - patch_width
            max_y = img_height - patch_height
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        else:
            x,y = 0,0
        image[:,y:y+patch_height,x:x+patch_width]=patch
        image = image.permute(1,2,0).numpy()
        return image

if __name__ == '__main__':
    from PIL import Image
    import torchvision
    random_patch_transform = RandomPatchTransform(device='cpu')
    im = Image.open("/Users/taowenwang/PycharmProjects/demo/openvla-main/0dataprocess/test.png")
    patch = torch.rand(3,32,32)
    # 加一个两个transform的代码
    mean = [torch.tensor([0.484375, 0.455078125, 0.40625]),torch.tensor([0.5,0.5,0.5])]
    std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]),torch.tensor([0.5,0.5,0.5])]

    for i in range(50):
        modified_images = random_patch_transform.apply_random_patch_batch([im],patch,mean=mean,std=std)
        vis_im = denormalize(modified_images[:,0:3,:,:],torch.tensor([0.484375, 0.455078125, 0.40625]),torch.tensor([0.228515625, 0.2236328125, 0.224609375])).squeeze(0)
        vis_im = torchvision.transforms.ToPILImage()(vis_im)
        vis_im.save(f"./test_function/{i}.png")