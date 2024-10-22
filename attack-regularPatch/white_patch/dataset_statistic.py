import numpy as np
from openvla_dataloader import get_bridge_dataloader
import torch
from transformers import AutoModelForVision2Seq
from appply_random_transform import RandomPatchTransform
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from transformers import AutoConfig
from tqdm import tqdm
import contextlib
from transformers import AutoModelForVision2Seq, AutoProcessor
import sys
import os

train_dataloader, val_dataloader = get_bridge_dataloader(batch_size=128)
dataloader = train_dataloader
import numpy as np
import matplotlib.pyplot as plt
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)
# 定义区间数量
num_bins = 256

# 创建 bins 范围，-1 到 1，划分成 256 个区间
bins = np.linspace(-1, 1, num_bins + 1)

# 初始化存储每个自由度的频数统计
histograms = np.zeros((7, num_bins))

# 假设你已经有了 dataloader，每次返回 [batch_size, 7] 的数据
idx = 0
for data in tqdm(dataloader):
    # 假设 batch 是一个 numpy 数组，形状为 [batch_size, 7]，如果不是，先转换为 numpy 数组
    label = data["labels"][:, 1:]
    mask = label > 31743
    continuous_actions_gt = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(label[mask].cpu().numpy())
    )
    temp_continuous_actions_gt = continuous_actions_gt.view(continuous_actions_gt.shape[0] // 7, 7)
    batch_actions = temp_continuous_actions_gt.numpy()  # 如果 batch 是 tensor，需转换为 numpy 数组
    # 对每个自由度进行统计，并逐批更新直方图
    for i in range(7):
        hist, _ = np.histogram(batch_actions[:, i], bins=bins)
        histograms[i] += hist  # 累加每个 batch 的统计结果
    if idx ==10000:
        break
    idx += 1
# 绘制柱状图
fig, axes = plt.subplots(7, 1, figsize=(10, 15))
for i in range(7):
    axes[i].bar(range(num_bins), histograms[i])

    # 找到统计量最大的 bin
    max_bin_index = np.argmax(histograms[i])
    max_bin_value = histograms[i][max_bin_index]

    # 在 x 轴上标记最大 bin
    axes[i].annotate(f'Max: {max_bin_index}/{max_bin_value}',
                     xy=(max_bin_index, max_bin_value),
                     xytext=(max_bin_index, max_bin_value + max_bin_value * 0.05),  # 提高标记位置
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     ha='center', color='red')

    axes[i].set_title(f'Action {i + 1}')
    axes[i].set_xlabel('Bin Index')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('action_histograms—small.png', dpi=300)
# plt.show()
