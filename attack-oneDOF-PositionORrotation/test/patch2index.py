import torch
from torchvision import transforms
def get_patch_emb_index(canvas):
    # 假设你的 canvas 是 [224, 224] 的张量
    # canvas = torch.randn(224, 224)  # 示例张量，实际应是你的 canvas 输入
    # canvas = canvas[0]

    # 1. 找到非 -100 的区域（即 patch 的区域）
    patch_mask = (canvas != -100)  # 得到一个布尔掩码，标记出 patch 区域
    # patch_mask 形状为 [224, 224]，True 的位置代表 patch 的区域

    # 2. 将 canvas 划分为 14x14 个 16x16 的网格（patch）
    patch_dim = 14
    image_size = 224
    grid_size = image_size // patch_dim  # 14

    # 3. 初始化一个空列表来存储受影响的 token 索引
    affected_tokens = set()
    to_pil = transforms.ToPILImage()
    # 4. 遍历每个 16x16 的 patch，检查是否被 patch 覆盖
    counter = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算当前 patch 在 canvas 中的范围
            x_start, y_start = i * patch_dim,j * patch_dim
            x_end, y_end = x_start + patch_dim, y_start + patch_dim
            small_patch = canvas[:,x_start:x_end, y_start:y_end]

            # 转换为 PIL 图像并保存
            # img = to_pil(small_patch)  # 转换为 [1, 16, 16] 的形状
            # img.save(f"patch_{counter}.png")  # 文件名为序号
            # 检查这个 16x16 的区域内是否有 patch 的内容（即非 -100 的值）
            if patch_mask[:, x_start:x_end, y_start:y_end].any():
                # 如果该区域有 patch 的影响，则记录对应的 token 索引
                token_index = i * grid_size + j
                affected_tokens.add(token_index)
            counter +=1
    # 将受影响的 token 索引转换为列表（可选）
    affected_tokens = list(affected_tokens)
    return affected_tokens
    # print("受 patch 影响的 token 索引:", affected_tokens)

if __name__ == '__main__':
    patch_path = "/spl_data/tw9146/openvla-main/run/white_patch_attack/73cc0c44-cd52-4850-803f-dd3eec888b35/500/patch.pt"
    patch = torch.load(patch_path)
    from PIL import Image
    im = Image.open("../../0dataprocess/test.png")
    import sys
    sys.path.append("../white_patch/")
    from appply_random_transform import RandomPatchTransform

    randomPatchTransform = RandomPatchTransform("cpu")
    mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
    std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
    modified_images, canvas_list = randomPatchTransform.paste_patch_fix([im], patch, mean=mean,
                                                                        std=std,inference=True)
    canvas_list[0][canvas_list==-100]=0
    import torchvision
    im = torchvision.transforms.ToPILImage()(canvas_list[0])
    im.save("mask.png")
    get_patch_emb_index(canvas_list[0])
    a=1
