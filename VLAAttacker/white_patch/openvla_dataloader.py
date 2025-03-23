import os
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor
from torch.utils.data import Dataset, DataLoader, Subset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import random
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_bridge_dataloader(batch_size,server):
    vla_path: str = "openvla/openvla-7b"
    data_root_dir = Path(f"{server}/openvla-main/dataset")
    dataset_name = "bridge_orig"
    shuffle_buffer_size = 100_000
    image_aug = False
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform( #
        action_tokenizer,
        processor.tokenizer,
        prompt_builder_fn=PurePromptBuilder if "v01" not in vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset_train = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=tuple([224,224]),
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
    )
    vla_dataset_val = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=tuple([224,224]),
        shuffle_buffer_size=shuffle_buffer_size,
        train=False,
        image_aug=image_aug,
    )

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(
        vla_dataset_train,
        batch_size=batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        # shuffle=True
    )
    val_dataloader = DataLoader(
        vla_dataset_val,
        batch_size=8,#32
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    return train_dataloader, val_dataloader


def get_dataloader(batch_size,server,dataset,vla_path):
    # vla_path: str = "openvla/openvla-7b"
    data_root_dir = Path(f"{server}/openvla-main/dataset")
    # dataset_name = "bridge_orig"
    if dataset == "bridge_orig":
        vla_path = "openvla/openvla-7b"
    elif dataset == "libero_spatial":
        vla_path = "openvla/openvla-7b-finetuned-libero-spatial"
        dataset += "_no_noops"
    elif dataset == "libero_object":
        vla_path = "openvla/openvla-7b-finetuned-libero-object"
        dataset += "_no_noops"
    elif dataset == "libero_goal":
        vla_path = "openvla/openvla-7b-finetuned-libero-goal"
        dataset += "_no_noops"
    elif dataset == "libero_10":
        vla_path = "openvla/openvla-7b-finetuned-libero-10"
        dataset += "_no_noops"
    else:
        assert False, "Invalid dataset"
    shuffle_buffer_size = 100_000
    image_aug = False
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform( #
        action_tokenizer,
        processor.tokenizer,
        prompt_builder_fn=PurePromptBuilder if "v01" not in vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset_train = RLDSDataset(
        data_root_dir,
        dataset,
        batch_transform,
        resize_resolution=tuple([224,224]),
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
    )
    vla_dataset_val = RLDSDataset(
        data_root_dir,
        dataset,
        batch_transform,
        resize_resolution=tuple([224,224]),
        shuffle_buffer_size=shuffle_buffer_size,
        train=False,
        image_aug=image_aug,
    )

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(
        vla_dataset_train,
        batch_size=batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        # shuffle=True
    )
    val_dataloader = DataLoader(
        vla_dataset_val,
        batch_size=8,#32
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    return train_dataloader, val_dataloader
