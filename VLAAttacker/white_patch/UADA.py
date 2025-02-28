import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithPast
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
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
        print(f"alpha: {self.alpha}, belta: {self.belta}")
        self.randomPatchTransform = RandomPatchTransform(self.vla.device,resize_patch)
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.optimizer = optimizer

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

        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.save_dir))
        plt.clf()
        torch.save(self.loss_buffer, '%s/loss' % (self.save_dir))

    def patchattack_unconstrained(self, train_dataloader, val_dataloader, num_iter=5000, target_action=np.zeros(7),
                                  patch_size=[3, 50, 50], lr=1 / 255, accumulate_steps=1, maskidx=[], warmup=20,
                                  filterGripTrainTo1=False, geometry=False,innerLoop=1,args=None):
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

            print("masking labels...")
            labels = self.mask_labels(labels, maskidx)

            for inner_loop in range(innerLoop):
                if geometry:
                    modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch,
                                                                                         mean=self.mean,
                                                                                         std=self.std,
                                                                                         geometry=geometry)
                output: CausalLMOutputWithPast = self.vla(
                    input_ids=input_ids.to(self.vla.device),
                    attention_mask=attention_mask.to(self.vla.device),
                    pixel_values=modified_images.to(torch.bfloat16).to(self.vla.device),
                    labels=labels,
                )
                celoss = output.loss
                loss, angle_loss, distance_loss = self.weighted_loss(output.logits, labels,maskidx)
                loss.backward()
                if self.optimizer == "adamW":
                    if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                        torch.nn.utils.clip_grad_norm_([patch], max_norm=5 * 6e-4*torch.tensor(patch.shape).prod().sqrt().item(), norm_type=2)
                        log_patch_grad = patch.grad.detach().mean().item()
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
            train_relative_distance = self.calculate_relative_distance(continuous_actions_pred,
                                                                 continuous_actions_gt, maskidx,
                                                                 train_relative_distance)
            train_logdata = {"TRAIN_attack_loss(CE)": celoss.item(),
                            "TRAIN_patch_gradient": log_patch_grad,
                            "TRAIN_LR": optimizer.param_groups[0]["lr"],
                            "TRAIN_ANGLE_LOSS": angle_loss,
                            "TRAIN_DISTANCE_LOSS":distance_loss}
            for key, value in train_relative_distance.items():
                property_name = f"train_rd_{key}"
                train_logdata[property_name] = sum(value) / len(value)
            if args.wandb_project != "false":
                wandb.log(train_logdata,step=i)
            self.train_CE_loss.append(loss.item())

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
                        if geometry:
                            modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values,
                                                                                                 patch,
                                                                                                 mean=self.mean,
                                                                                                 std=self.std,
                                                                                                 geometry=geometry)
                        labels = self.mask_labels(labels, maskidx)
                        output: CausalLMOutputWithPast = self.vla(
                            input_ids=input_ids.to(self.vla.device),
                            attention_mask=attention_mask.to(self.vla.device),
                            pixel_values=modified_images.to(torch.bfloat16).to(self.vla.device),
                            labels=labels,
                        )
                        val_loss, val_angle_loss, val_distance_loss = self.weighted_loss(output.logits, labels,maskidx)
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
                    log_data["val_reverse_direction_loss"] = avg_reserve_loss
                    log_data["val_angle_loss"] = avg_angle_loss
                    log_data["val_distance_loss"] = avg_distance_loss
                    for key, value in relative_distance.items():
                        property_name = f"val_rd_{key}"
                        log_data[property_name] = sum(value) / len(value)
                        avg_relative_distance += sum(value) / len(value)
                    log_data["avg_relative_distance"] = avg_relative_distance
                    if args.wandb_project != "false":
                        wandb.log(log_data,step=i)
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
                        if args.wandb_project != "false":
                            wandb.log({"AdvImg": [wandb.Image(pil_img) for pil_img in pil_imgs]})
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
                self.save_info(path=self.save_dir)
                torch.cuda.empty_cache()

    def modifiy_labels(self, labels,
                       target_action={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8}):
        newlabels = []
        for j in range(labels.shape[0]):
            temp_label = labels[j]
            first_valid_index = (temp_label != -100).nonzero(as_tuple=True)[0].item()
            for key, value in target_action.items():
                if value != -100:
                    temp_label[int(first_valid_index + int(key))] = value
            newlabels.append(temp_label.unsqueeze(0))
            print(temp_label)
        newlabels = torch.cat(newlabels, dim=0)
        return newlabels

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
        pred = pred.clone().view(pred.shape[0]//len(maskidx),len(maskidx))
        gt = gt.clone().view(gt.shape[0]//len(maskidx),len(maskidx))
        for idx1 in range(pred.shape[0]):
            for idx2 in range(pred.shape[1]):
                anchor = gt[idx1,idx2]
                input_point = pred[idx1,idx2]
                upper_bound = 1
                lower_bound = -1
                distance_to_upper = upper_bound - anchor
                distance_to_lower = anchor - lower_bound
                max_boundary_distance = max(distance_to_upper, distance_to_lower)
                distance_to_anchor = abs(input_point - anchor)
                temp_relative_distance = distance_to_anchor / max_boundary_distance
                relative_distance[f"{str(maskidx[idx2])}"].append(temp_relative_distance.item())
        return relative_distance


    def weighted_loss(self, logits, labels,maskid):
        temp_label = labels[:,1:].to(labels.device) # (bs,seq_len)
        action_mask = temp_label!= -100
        temp_logits = logits[:,:,31744:32000] # (bs,seq_len,256) only consider the 256 classes for the target class
        action_logits = temp_logits[:,self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
        reweigh = torch.arange(1,257).to(logits.device) # [1,2,3...,256]
        temp_prob = F.softmax(action_logits,dim=-1) # [bs,seq_len,256]
        reweighted_prob = (temp_prob * reweigh).sum(dim=-1)  # [bs,seq_len]
        xyz_reweigthed = torch.cat([label[action_mask[i]].unsqueeze(0) for i, label in enumerate(reweighted_prob)],dim=0)
        xyz_reweigthed = torch.cat([xyz_reweigthed[:,i].unsqueeze(1) for i in maskid],dim=1)
        xyz_label = torch.cat([label[action_mask[i]].unsqueeze(0) for i, label in enumerate(temp_label)],dim=0)-31743
        xyz_label = torch.cat([xyz_label[:,i].unsqueeze(1) for i in maskid],dim=1)
        xyz_reweigthed = (xyz_reweigthed-1)/(255)
        xyz_label = (xyz_label-1)/(255)

        cosine_sim = F.cosine_similarity(xyz_reweigthed, xyz_label, dim=1)
        angle_loss = (cosine_sim + 1).mean()
        relative_distance = 0
        for x in range(xyz_reweigthed.shape[0]):
            for y in range(xyz_reweigthed.shape[1]):
                upper_bound = 1
                lower_bound = 0
                distance_to_upper = upper_bound - xyz_label[x,y]
                distance_to_lower = xyz_label[x,y] - lower_bound
                max_boundary_distance = max(distance_to_upper, distance_to_lower)
                distance_to_anchor = abs(xyz_reweigthed[x,y] - xyz_label[x,y])
                relative_distance += distance_to_anchor / max_boundary_distance
        UAD = 1/(relative_distance+1e-3)


        total_loss = self.alpha*angle_loss + self.belta*UAD
        return total_loss, angle_loss.item(), UAD.item()
        
    def mask_labels(self, labels, maskidx):
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



