# image processor
# resize-naive - 224 224 interpolations - bicubic
import time

import torch
import torchvision
from tensorflow.python.ops.numpy_ops.np_utils import result_type_unary
from tqdm import tqdm
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import AutoProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import torchvision.transforms.functional as TVF
from torchvision import transforms
import random
import torch.nn.functional as F
from appply_random_transform import RandomPatchTransform
from tqdm import tqdm
import os
import transformers
import pickle
IGNORE_INDEX = -100

def normalize(images,mean,std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images,mean,std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class OpenVLAAttacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="pgd",resize_patch=False,alpha=0.5,belta=0.5):
        self.vla = vla.eval()
        self.vla.vision_backbone_requires_grad = True
        # self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        self.image_transform = self.processor.image_processor.apply_transform
        self.base_tokenizer = self.processor.tokenizer
        self.predict_stop_token: bool = True
        self.pad_token_id = 32000
        self.model_max_length = 2048
        self.loss_buffer = []
        self.save_dir = save_dir
        self.adv_action_L1_loss = []
        self.avg_angle_loss = []
        self.avg_distance_loss = []
        self.avg_reserve_loss = []
        self.min_val_avg_CE_loss = 1000000
        self.min_val_avg_L1_loss = 1000000
        self.max_relative_distance = -1000000
        self.reverse_direction_loss = 100000
        self.alpha = alpha
        self.belta = belta
        self.randomPatchTransform = RandomPatchTransform(self.vla.device,resize_patch)
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.optimizer = optimizer

        # im configuration
        self.input_sizes = [[3, 224, 224], [3, 224, 224]]
        self.tvf_resize_params = [
            {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]},
            {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]}
        ]
        self.tvf_crop_params = [
            {'output_size': [224, 224]},
            {'output_size': [224, 224]}
        ]
        self.tvf_normalize_params = [
            {'inplace': False, 'mean': [0.484375, 0.455078125, 0.40625],
             'std': [0.228515625, 0.2236328125, 0.224609375]},
            {'inplace': False, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        ]

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.save_dir))
        plt.clf()
        torch.save(self.loss_buffer, '%s/loss' % (self.save_dir))

    def wrapperfunction(self, prompt, img, action=np.zeros(7)):
        """
        prompt: string prompt without template
        """

        # process img PIL -> tensor (1,6,224,224)
        pixel_values = self.image_transform(img)
        # prismatic / extern / hf / processing_prismatic.py - line 135

        # process prompt
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {prompt}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        print(f"Conversation: {conversation}; action: {action}")
        for turn in conversation:
            self.prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = self.base_tokenizer(self.prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        input_ids = pad_sequence([input_ids], batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence([labels], batch_first=True, padding_value=IGNORE_INDEX)

        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output

    def im_transform(self, img):
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)
        # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0
        img_t = torch.vstack(imgs_t)
        return img_t

    def attack_unconstrained(self, prompt, img, num_iter=2000, action=np.zeros(7), alpha=1 / 255):
        data = self.wrapperfunction(prompt, img, action)
        adv_noise = torch.rand((1, 3, 224, 224)).to(self.vla.device, dtype=torch.bfloat16)
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for i in tqdm(range(num_iter)):
            # transform 1 -> A
            adv_noise1 = normalize(adv_noise,
                                   mean=torch.tensor([0.484375, 0.455078125, 0.40625]).to(self.vla.device),
                                   std=torch.tensor([0.228515625, 0.2236328125, 0.224609375]).to(self.vla.device))
            # transform 2 -> B
            adv_noise2 = normalize(adv_noise, mean=torch.tensor([0.5, 0.5, 0.5]).to(self.vla.device),
                                   std=torch.tensor([0.5, 0.5, 0.5]).to(self.vla.device))
            pixel_values = torch.cat([adv_noise1, adv_noise2], dim=1)
            data["pixel_values"] = pixel_values
            output: CausalLMOutputWithPast = self.vla(
                input_ids=data["input_ids"].to(self.vla.device),
                attention_mask=data["attention_mask"].to(self.vla.device),
                pixel_values=data["pixel_values"].to(torch.bfloat16).to(self.vla.device),
                labels=data["labels"],
            )
            loss = output.loss
            loss.backward()
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.vla.zero_grad()
            self.loss_buffer.append(loss.item())
            print("target_loss: %f" % (loss.item()))
            wandb.log(
                {
                    "attack_loss(CE)": loss.item(),
                },
                step=i,
            )
            if i % 20 == 0:
                self.plot_loss()

            if i % 100 == 0:
                print('######### Output - Iter = %d ##########' % i)
                x_adv1 = normalize(adv_noise,
                                   mean=torch.tensor([0.484375, 0.455078125, 0.40625]).to(self.vla.device),
                                   std=torch.tensor([0.228515625, 0.2236328125, 0.224609375]).to(self.vla.device))
                # transform 2 -> B
                x_adv2 = normalize(adv_noise, mean=torch.tensor([0.5, 0.5, 0.5]).to(self.vla.device),
                                   std=torch.tensor([0.5, 0.5, 0.5]).to(self.vla.device))
                x_adv = torch.cat([x_adv1, x_adv2], dim=1)
                data["pixel_values"] = x_adv
                output: CausalLMOutputWithPast = self.vla(
                    input_ids=data["input_ids"].to(self.vla.device),
                    attention_mask=data["attention_mask"].to(self.vla.device),
                    pixel_values=data["pixel_values"].to(torch.bfloat16).to(self.vla.device),
                    labels=data["labels"],
                )
                # Compute Accuracy and L1 Loss for Logging
                # action_logits = output.logits[:, self.vla.module.vision_backbone.featurizer.patch_embed.num_patches: -1]
                action_logits = output.logits[:, self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = data["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > self.action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                # correct_preds = (action_preds == action_gt) & mask
                # action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                self.adv_action_L1_loss.append(action_l1_loss.item())
                wandb.log(
                    {
                        "action_L1_loss": action_l1_loss.item(),
                    },
                    step=i,
                )
                # save
                if action_l1_loss.item() < self.min_l1_loss:
                    self.min_l1_loss = action_l1_loss.item()
                    torch.save(adv_noise.detach().cpu(), '%s/adv_noise_iter_%d.pt' % (self.save_dir, i))
                    pil_img = TVF.to_pil_image(adv_noise.detach().cpu().squeeze(0))
                    pil_img.save('%s/adv_noise_iter_%d.png' % (self.save_dir, i))
                    wandb.log({"AdvImg": [wandb.Image(pil_img)]})

    def patchattack_unconstrained(self, train_dataloader, val_dataloader, num_iter=5000, target_action=np.zeros(7),
                                  patch_size=[3, 50, 50], lr=1 / 255, accumulate_steps=1, maskidx=[], warmup=20,
                                  filterGripTrainTo1=False, geometry=False, colorjitter=False,guide=False,innerLoop=1,reverse_direction=True):
        # log record
        self.val_CE_loss = []
        self.val_L1_loss = []
        self.val_ASR = []
        self.train_CE_loss = []
        self.val_relative_distance = []
        angle_loss=0
        distance_loss = 0
        log_patch_grad = 0

        patch = torch.rand(patch_size).to(self.vla.device)
        patch.requires_grad_(True)
        patch.retain_grad()
        if self.optimizer == "adamW":
            optimizer = transformers.AdamW([patch], lr=lr)
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=int(num_iter / accumulate_steps),
                num_cycles=0.5,
                last_epoch=-1,
            )
        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader)
        for i in tqdm(range(num_iter)):
            train_relative_distance = {f"{idx}": [] for idx in maskidx}
            data = next(train_iterator)
            if len(maskidx) == 1 and maskidx[0] == 6 and filterGripTrainTo1:
                labels, attention_mask, input_ids, pixel_values = self.filter_train(data)
            else:
                pixel_values = data["pixel_values"]
                labels = data["labels"].to(self.vla.device)
                attention_mask = data["attention_mask"].to(self.vla.device)
                input_ids = data["input_ids"].to(self.vla.device)
            # if not reverse_direction:
            #     print("masking labels...")
            #     labels = self.mask_labels(labels, maskidx)

            for inner_loop in range(innerLoop):
                if not geometry and not colorjitter:
                    modified_images = self.randomPatchTransform.paste_patch_fix(pixel_values, patch, mean=self.mean,
                                                                                std=self.std)
                else:
                    modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch,
                                                                                         mean=self.mean,
                                                                                         std=self.std,
                                                                                         geometry=geometry,
                                                                                         colorjitter=colorjitter)
                output: CausalLMOutputWithPast = self.vla(
                    input_ids=input_ids.to(self.vla.device),
                    attention_mask=attention_mask.to(self.vla.device),
                    pixel_values=modified_images.to(torch.bfloat16).to(self.vla.device),
                    labels=labels,
                )
                loss, angle_loss, distance_loss = self.weighted_loss(output.logits, labels)
                loss.backward()
                log_patch_grad = patch.grad.detach().mean().item()
                if self.optimizer == "adamW":
                    if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                        optimizer.step()
                        patch.data = patch.data.clamp(0, 1)
                        optimizer.zero_grad()
                        self.vla.zero_grad()
                        torch.cuda.empty_cache()
                elif self.optimizer == "pgd":
                    if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                        loss.backward()
                        log_patch_grad = patch.grad.detach().mean().item()
                        patch.data = (patch.data - lr * patch.grad.detach().sign()).clamp(0, 1)
                        self.vla.zero_grad()
                        patch.grad.zero_()

            if self.optimizer == "adamW":
                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                    scheduler.step()
            action_logits = output.logits[:,
                            self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = labels[:, 1:].to(action_preds.device)
            mask = action_gt > self.action_tokenizer.action_token_begin_idx
            continuous_actions_pred = torch.tensor(
                self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            # calculate relative distance metric
            train_relative_distance = self.calculate_relative_distance(continuous_actions_pred,
                                                                 continuous_actions_gt, maskidx,
                                                                 train_relative_distance)
            train_logdata = {"TRAIN_attack_loss(CE)": loss.item(),
                            "TRAIN_patch_gradient": log_patch_grad,
                            "TRAIN_LR": optimizer.param_groups[0]["lr"],
                            "TRAIN_ANGLE_LOSS": angle_loss,
                            "TRAIN_DISTANCE_LOSS":distance_loss}
            for key, value in train_relative_distance.items():
                property_name = f"train_rd_{key}"
                train_logdata[property_name] = sum(value) / len(value)
            wandb.log(train_logdata,step=i)
            self.train_CE_loss.append(loss.item())
            # save here

            if i % 100 == 0:
                self.plot_loss()

            if i % 100 == 0:
                avg_CE_loss = 0
                avg_L1_loss = 0
                val_num_sample = 0
                success_attack_num = 0
                relative_distance = {f"{idx}": [] for idx in maskidx}
                avg_relative_distance = 0
                avg_angle_loss = 0
                avg_distance_loss = 0
                avg_reserve_loss = 0
                print("evaluating...")
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for j in tqdm(range(100)):
                        try:
                            data = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(val_dataloader)
                            data = next(val_iterator)
                        pixel_values = data["pixel_values"]
                        labels = data["labels"].to(self.vla.device)
                        attention_mask = data["attention_mask"].to(self.vla.device)
                        input_ids = data["input_ids"].to(self.vla.device)
                        val_num_sample += labels.shape[0]
                        if not geometry and not colorjitter:
                            modified_images = self.randomPatchTransform.paste_patch_fix(pixel_values, patch,
                                                                                        mean=self.mean,
                                                                                        std=self.std)
                        else:
                            modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values,
                                                                                                 patch,
                                                                                                 mean=self.mean,
                                                                                                 std=self.std,
                                                                                                 geometry=geometry,
                                                                                                 colorjitter=colorjitter)
                        if not reverse_direction:
                            labels = self.mask_labels(labels, maskidx)
                        output: CausalLMOutputWithPast = self.vla(
                            input_ids=input_ids.to(self.vla.device),
                            attention_mask=attention_mask.to(self.vla.device),
                            pixel_values=modified_images.to(torch.bfloat16).to(self.vla.device),
                            labels=labels,
                        )
                        val_loss, val_angle_loss, val_distance_loss = self.weighted_loss(output.logits, labels)
                        action_logits = output.logits[:,
                                        self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
                        action_preds = action_logits.argmax(dim=2)
                        action_gt = labels[:, 1:].to(action_preds.device)
                        mask = action_gt > self.action_tokenizer.action_token_begin_idx
                        continuous_actions_pred = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                        )
                        continuous_actions_gt = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                        )
                        # calculate relative distance metric
                        relative_distance = self.calculate_relative_distance(continuous_actions_pred,
                                                                             continuous_actions_gt, maskidx,
                                                                             relative_distance)
                        avg_angle_loss += val_angle_loss
                        avg_distance_loss += val_distance_loss
                        avg_reserve_loss += val_loss
                    torch.cuda.empty_cache()
                    avg_angle_loss /= val_num_sample
                    avg_distance_loss /= val_num_sample
                    avg_reserve_loss /= val_num_sample
                    log_data={}
                    log_data["reverse_direction_loss"] = avg_reserve_loss
                    log_data["avg_angle_loss"] = avg_angle_loss
                    log_data["avg_distance_loss"] = avg_distance_loss
                    for key, value in relative_distance.items():
                        property_name = f"val_rd_{key}"
                        log_data[property_name] = sum(value) / len(value)
                        avg_relative_distance += sum(value) / len(value)
                    wandb.log(log_data,step=i)
                    # save
                    if avg_reserve_loss.item() < self.reverse_direction_loss:
                        self.reverse_direction_loss = avg_reserve_loss.item()
                        temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                        os.makedirs(temp_save_dir, exist_ok=True)
                        torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                        val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                        os.makedirs(val_related_file_path, exist_ok=True)
                        modified_images = self.randomPatchTransform.denormalize(
                            modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                        pil_imgs = []
                        for o in range(modified_images.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(modified_images[o, :, :, :])
                            pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                            pil_imgs.append(pil_img)
                        wandb.log({"AdvImg": [wandb.Image(pil_img) for pil_img in pil_imgs]})
                    # save last checkpoint
                    temp_save_dir = os.path.join(self.save_dir, "last")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    modified_images = self.randomPatchTransform.denormalize(
                        modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                self.val_CE_loss.append(avg_CE_loss)
                self.val_L1_loss.append(avg_L1_loss)
                self.val_ASR.append(success_attack_num / val_num_sample)
                self.avg_angle_loss.append(avg_angle_loss / val_num_sample)
                self.avg_distance_loss.append(avg_distance_loss / val_num_sample)
                self.avg_reserve_loss.append(avg_reserve_loss / val_num_sample)
                # save here
                self.save_info(path=self.save_dir)
                torch.cuda.empty_cache()  # torch.save(modified_images.detach().cpu(), os.path.join(val_related_file_path,"modified_images.pt"))

    def modifiy_labels(self, labels,
                       target_action={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8}):
        newlabels = []
        for j in range(labels.shape[0]):
            temp_label = labels[j]
            # temp_label[temp_label != -100] = target_action
            first_valid_index = (temp_label != -100).nonzero(as_tuple=True)[0].item()
            for key, value in target_action.items():
                if value != -100:
                    temp_label[int(first_valid_index + int(key))] = value
            newlabels.append(temp_label.unsqueeze(0))
            print(temp_label)
        newlabels = torch.cat(newlabels, dim=0)
        return newlabels

    def calculate_ASR_by_DoE(self, pred, gt):
        assert pred.shape == gt.shape
        temp_doe_ASR = [0, 0, 0, 0, 0, 0, 0]
        for pred_idx in range(pred.shape[0]):
            for DoE_idx in range(pred.shape[1]):
                if pred[pred_idx, DoE_idx] == gt[pred_idx, DoE_idx]:
                    temp_doe_ASR[DoE_idx] += 1
        return temp_doe_ASR

    def calculate_01_ASR(self, pred, gt):
        temp_02other_success = 0
        gt_0_num = 0
        temp_12other_success = 0
        gt_1_num = 0
        gt_others_num = 0
        temp_other20_success = 0
        print(f"calculate_01_ASR:{gt}")
        for idx in range(gt.shape[0]):
            if gt[idx] == 31872:  # gt is 0
                gt_0_num += 1
                if pred[idx] != 31872:
                    temp_02other_success += 1
            elif gt[idx] == 31744:  # gt is 1
                gt_1_num += 1
                if pred[idx] != 31744:
                    temp_12other_success += 1
            elif gt[idx] != 31872:  # gt is not 0
                gt_others_num += 1
                if pred[idx] == 31872:
                    temp_other20_success += 1

        return temp_02other_success, gt_0_num, temp_12other_success, gt_1_num, temp_other20_success, gt_others_num

    def filter_train(self, data):
        pixel_values = data["pixel_values"]
        labels = data["labels"].to(self.vla.device)
        attention_mask = data["attention_mask"].to(self.vla.device)
        input_ids = data["input_ids"].to(self.vla.device)

        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        one_index = []
        for idx in range(masked_labels.shape[0]):
            if masked_labels[idx, 6] == 31744:
                one_index.append(idx)
        if 1 < len(one_index) < 8:
            labels = labels[one_index, :]
            attention_mask = attention_mask[one_index, :]
            input_ids = input_ids[one_index, :]
            pixel_values = [pixel_values[i] for i in one_index]
        elif len(one_index) > 8:
            chosen = random.sample(one_index, k=8)
            labels = labels[chosen, :]
            attention_mask = attention_mask[chosen, :]
            input_ids = input_ids[chosen, :]
            pixel_values = [pixel_values[i] for i in chosen]
        elif one_index is None:
            chosen = random.sample(range(labels.shape[0]), k=8)
            labels = labels[chosen, :]
            attention_mask = attention_mask[chosen, :]
            input_ids = input_ids[chosen, :]
            pixel_values = [pixel_values[i] for i in chosen]
        # print(f"one_index: {one_index}, labels.shape: {labels.shape}")
        return labels, attention_mask, input_ids, pixel_values

    def save_info(self, path):
        with open(os.path.join(path, 'val_relative_distance.pkl'), 'wb') as file:
            pickle.dump(self.val_relative_distance, file)
        with open(os.path.join(path, 'val_CE_loss.pkl'), 'wb') as file:
            pickle.dump(self.val_CE_loss, file)
        with open(os.path.join(path, 'val_L1_loss.pkl'), 'wb') as file:
            pickle.dump(self.val_L1_loss, file)
        with open(os.path.join(path, 'val_ASR.pkl'), 'wb') as file:
            pickle.dump(self.val_ASR, file)
        with open(os.path.join(path, 'train_CE_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_CE_loss, file)
        with open(os.path.join(path, 'val_avg_angle_loss.pkl'), 'wb') as file:
            pickle.dump(self.avg_angle_loss, file)
        with open(os.path.join(path, 'val_avg_distance_loss.pkl'), 'wb') as file:
            pickle.dump(self.avg_distance_loss, file)
        with open(os.path.join(path, 'val_avg_reserve_loss.pkl'), 'wb') as file:
            pickle.dump(self.avg_reserve_loss, file)


    def calculate_relative_distance(self,pred,gt,maskidx,relative_distance):
        # process input -> (n,) -> (a,b)
        pred = pred.clone().view(pred.shape[0]//len(maskidx),len(maskidx))
        gt = gt.clone().view(gt.shape[0]//len(maskidx),len(maskidx))
        for idx1 in range(pred.shape[0]):
            for idx2 in range(pred.shape[1]):
                anchor = gt[idx1,idx2]
                input_point = pred[idx1,idx2]
                # define the upper and lower bound of the input range
                upper_bound = 1
                lower_bound = -1
                # cal the distance to upper and lower bound for anchor
                distance_to_upper = upper_bound - anchor
                distance_to_lower = anchor - lower_bound
                # choose the larger distance as the max_boundary_distance
                max_boundary_distance = max(distance_to_upper, distance_to_lower)
                # cal the distance of input to anchor
                distance_to_anchor = abs(input_point - anchor)
                # cal relative distance
                temp_relative_distance = distance_to_anchor / max_boundary_distance
                relative_distance[f"{str(maskidx[idx2])}"].append(temp_relative_distance.item())
        return relative_distance

    def mask_labels(self,labels,maskidx):
        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        for idx in range(7):
            if idx not in maskidx:
                masked_labels[:, idx] = -100
        newlabels = []
        for j in range(labels.shape[0]):
            temp_label = labels[j]
            temp_label[temp_label > 2] = masked_labels[j]
            newlabels.append(temp_label.unsqueeze(0))
        return torch.cat(newlabels, dim=0)

    def change_target(self,gt):
        mask= gt!=-100
        random_assign = torch.randint(0, 2, gt[mask & (gt == 31872)].shape, dtype=torch.bool).to(gt.device)
        gt[mask & (gt == 31872)] = torch.where(random_assign, torch.tensor(31744, dtype=gt.dtype,device=gt.device),torch.tensor(31999, dtype=gt.dtype,device=gt.device))
        gt[mask & (gt>31872)] = 31744
        gt[mask & (gt<31872)] = 31999
        return gt


    def weighted_loss(self, logits, labels):
        # logits.shape = (bs,seq_len+1,32064)
        temp_label = labels[:,1:].to(labels.device) # (bs,seq_len)
        action_mask = temp_label!= -100
        temp_logits = logits[:,:,31744:32000] # (bs,seq_len,256) only consider the 256 classes for the target class
        action_logits = temp_logits[:,self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
        reweigh = torch.arange(1,257).to(logits.device) # [1,2,3...,256]
        # reweigh = 1-reweigh/128 # assign
        temp_prob = F.softmax(action_logits,dim=-1) # [bs,seq_len,256]
        reweighted_prob = (temp_prob * reweigh).sum(dim=-1)  # [bs,seq_len]
        # get position xyz
        xyz_reweigthed = torch.cat([label[action_mask[i]].unsqueeze(0) for i, label in enumerate(reweighted_prob)],dim=0)
        xyz_reweigthed = xyz_reweigthed[:,:3]
        # get xyz_label
        xyz_label = torch.cat([label[action_mask[i]].unsqueeze(0) for i, label in enumerate(temp_label)],dim=0)-31743
        xyz_label = xyz_label[:,:3]
        # rescale
        xyz_reweigthed = (xyz_reweigthed-1)/(255)
        xyz_label = (xyz_label-1)/(255)

        # calculate loss 1  angle
        cosine_sim = F.cosine_similarity(xyz_reweigthed, xyz_label, dim=1)
        angle_loss = (cosine_sim + 1).mean()
        # calculate loss 2  distance loss
        distance_loss = 1/(torch.norm(xyz_reweigthed - xyz_label, p=2, dim=1).mean()+1e-3)

        total_loss = self.alpha*angle_loss + self.belta*distance_loss
        return total_loss, angle_loss.item(), distance_loss.item()



