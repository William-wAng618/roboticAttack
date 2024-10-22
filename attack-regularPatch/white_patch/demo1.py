def apply_random_patch_batch(self, images, patch, mean, std):
    """
    random paste patch to images

    param:
    images (torch.Tensor):  list PIL images
    patch (torch.Tensor):  [3, patch_height, patch_width] patch

    return:
    torch.Tensor: batch img with patch added
    """

    # 获取图像和补丁的尺寸
    patch_channels, patch_height, patch_width = patch.shape

    modified_images = []
    # apply patch to each image in the batch
    for i in range(len(images)):
        # process org image, with no transform
        temp_im = transforms.ToTensor()(images[i]).to(self.device)
        if temp_im.ndim == 3:
            temp_im = temp_im.unsqueeze(0)
        temp_im = self.normalize(temp_im, mean=mean.to(self.device), std=std.to(self.device))
        batch_size, img_channels, img_height, img_width = temp_im.shape

        # process patch, with transform
        scale = random.uniform(0.5, 1.5)
        height, width = int(patch_height * scale), int(patch_width * scale)  # random scale patch
        aug_patch = transforms.Resize((height, width))(patch)
        aug_patch = self.normalize(aug_patch, mean=mean.to(self.device), std=std.to(self.device))

        # color jitter transform
        aug_patch = self.random_color_jitter(aug_patch, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                                             saturation_range=(0.8, 1.2),
                                             hue_range=(-0.1, 0.1))

        # Geometry transform
        affline_matrix = self.combined_transform_matrix()
        aug_patch = self.apply_affine_transform(aug_patch, affline_matrix)

        _, aug_patch_channels, aug_patch_height, aug_patch_width = aug_patch.shape

        # 计算补丁可以放置的最大 x 和 y 位置
        max_x = img_width - patch_width
        max_y = img_height - patch_height

        # 随机选择补丁的起始位置
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # 将补丁粘贴到随机位置
        temp_im[i, :, y:y + patch_height, x:x + patch_width] = patch
        modified_images.append(temp_im)
        # modified_images[i, :, y:y + patch_height, x:x + patch_width] = patch

    return torch.cat(modified_images, dim=0)


def random_color_jitter(self,image, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        """
        Random color dithering, including brightness, contrast, saturation and hue adjustments.

        Args:
            image (Tensor):  [C, H, W]
            brightness_range (tuple):  (min_brightness, max_brightness)
            contrast_range (tuple):  (min_contrast, max_contrast)
            saturation_range (tuple):  (min_saturation, max_saturation)
            hue_range (tuple):  (min_hue, max_hue)

        Returns:
            Tensor: Adjusted image
        """


        # 亮度调整
        image = image * brightness_factor

        # 对比度调整
        mean = image.mean(dim=[1, 2], keepdim=True)
        image = (image - mean) * contrast_factor + mean

        # 饱和度调整
        gray = image.mean(dim=0, keepdim=True)
        image = (image - gray) * saturation_factor + gray

        # 色调调整
        hue = torch.tensor(hue_factor)  # 转换为 Tensor
        hue = torch.clamp(hue, -0.5, 0.5)  # 使用 Number 类型参数
        image = (image + hue) % 1.0

        return image