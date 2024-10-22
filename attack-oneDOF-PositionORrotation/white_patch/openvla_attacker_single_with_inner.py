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
    def __init__(self, vla, processor, save_dir="",optimizer="pgd",resize_patch=False):
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
        self.min_val_avg_CE_loss = 1000000
        self.min_val_avg_L1_loss = 1000000
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
        imgs_t=[]
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)
        # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0
        img_t = torch.vstack(imgs_t)
        return img_t

    def attack_unconstrained(self, prompt, img, num_iter=2000,action=np.zeros(7), alpha=1/255):
        data = self.wrapperfunction(prompt, img, action)
        adv_noise = torch.rand((1, 3, 224, 224)).to(self.vla.device,dtype=torch.bfloat16)
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for i in tqdm(range(num_iter)):
            # transform 1 -> A
            adv_noise1 = normalize(adv_noise, mean=torch.tensor([0.484375, 0.455078125, 0.40625]).to(self.vla.device), std=torch.tensor([0.228515625, 0.2236328125, 0.224609375]).to(self.vla.device))
            # transform 2 -> B
            adv_noise2 = normalize(adv_noise, mean=torch.tensor([0.5, 0.5, 0.5]).to(self.vla.device), std=torch.tensor([0.5, 0.5, 0.5]).to(self.vla.device))
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
                x_adv1 =normalize(adv_noise,
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


    def patchattack_unconstrained(self,train_dataloader,val_dataloader,num_iter=5000,target_action=np.zeros(7),patch_size=[3,50,50],alpha=1/255, accumulate_steps=1,maskidx=[],warmup=20,filterGripTrainTo1=False,geometry=False, colorjitter=False,innerLoop=1):
        self.val_CE_loss = []
        self.val_L1_loss = []
        self.val_ASR = []
        self.val_inner_relatived_distance=[]
        self.train_CE_loss = []
        self.train_inner_avg_loss = []
        self.train_inner_relatived_distance=[]
        patch = torch.rand(patch_size).to(self.vla.device)
        patch.requires_grad_(True)
        patch.retain_grad()
        # target_action_org = torch.from_numpy(target_action).to(self.vla.device)
        target_action = self.base_tokenizer(self.action_tokenizer(target_action)).input_ids[2:]
        target_action.append(2)
        target_action = list(target_action)
        target_action = torch.tensor(target_action).to(self.vla.device)
        target_action[:-1] = - 100
        for item in maskidx:
            target_action[item] = 31872
        print(f"target_action: {target_action}")
        if self.optimizer == "adamW":
            optimizer = transformers.AdamW([patch], lr=alpha)
            # scheduler = transformers.get_scheduler(
            #     name="cosine",
            #     optimizer=optimizer,
            #     num_warmup_steps=5,
            #     num_training_steps=num_iter
            # )
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=int(num_iter/accumulate_steps),
                num_cycles=0.5,
                last_epoch=-1,
            )

        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader)
        for i in tqdm(range(num_iter)):
            torch.cuda.empty_cache()
            data = next(train_iterator)
            if len(maskidx)==1 and maskidx[0]==6 and filterGripTrainTo1:
                labels, attention_mask, input_ids, pixel_values = self.filter_train(data)
            else:
                pixel_values = data["pixel_values"]
                labels = data["labels"].to(self.vla.device)
                attention_mask = data["attention_mask"].to(self.vla.device)
                input_ids = data["input_ids"].to(self.vla.device)


            # process label
            newlabels = []
            # zero_num = 0
            for j in range(labels.shape[0]):
                temp_label = labels[j].clone()
                temp_label[temp_label != -100] = target_action
                newlabels.append(temp_label.unsqueeze(0))
                # indices = torch.where(labels[j] == 2)
                # if labels[j][indices[0].item()-1]==31872:
                #     zero_num += 1
                # print(temp_label)
            newlabels = torch.cat(newlabels, dim=0)
            inner_avg_loss = 0
            inner_relatived_distance = 0
            for inner_loop in range(innerLoop):
                # process img
                # modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch, mean=self.mean, std=self.std)
                # modified_images = self.randomPatchTransform.random_paste_patch(pixel_values, patch, mean=self.mean, std=self.std)
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
                    labels=newlabels,
                )
                loss = output.loss / accumulate_steps
                inner_avg_loss += loss.item()

                action_logits = output.logits[:,self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
                action_preds = action_logits.argmax(dim=2)
                temp_label = newlabels[:, 1:].clone()
                mask = temp_label > self.action_tokenizer.action_token_begin_idx
                continuous_actions_gt = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(temp_label[mask].cpu().numpy())
                )
                continuous_actions_pred = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                inner_relatived_distance+=self.calculate_relative_distance_target(continuous_actions_pred,continuous_actions_gt)
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
                        patch.data = (patch.data - alpha * patch.grad.detach().sign()).clamp(0, 1)
                        self.vla.zero_grad()
                        patch.grad.zero_()
            inner_avg_loss /= innerLoop
            inner_relatived_distance/=innerLoop

            if self.optimizer == "adamW":
                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                    scheduler.step()

            # patch.grad.zero_()
            # self.vla.zero_grad()
            # zero_precentage = zero_num / labels.shape[0]
            # one_precentage = 1 - zero_precentage
            self.loss_buffer.append(loss.item())
            print(f"target_loss: {loss.item()}")
            wandb.log(
                {
                    "TRAIN_attack_loss(CE)": loss.item(),
                    "TRAIN_patch_gradient": log_patch_grad,
                    "TRAIN_LR": optimizer.param_groups[0]["lr"],
                    "TRAIN_inner_avg_loss": inner_avg_loss,
                    "TRAIN_inner_relatived_distance": inner_relatived_distance,
                    # "TRAIN_one_precentage": one_precentage,
                },
                step=i,
            )
            self.train_CE_loss.append(loss.item())
            self.train_inner_avg_loss.append(inner_avg_loss)
            self.train_inner_relatived_distance.append(inner_relatived_distance)
            if i % 100 == 0:
                self.plot_loss()

            # if i % 500 == 0:
            if i % 100 == 0:
                avg_CE_loss = 0
                avg_L1_loss = 0
                val_num_sample = 0
                success_attack_num = 0
                val_inner_relatived_distance=0
                all_02other_success, all_gt_0_num, all_12other_success, all_gt_1_num, all_other20_success, all_gt_others_num =0,0,0,0,0,0
                print("evaluating...")
                with torch.no_grad():
                    # for j in tqdm(range(500)):
                    for j in tqdm(range(100)):
                    # for data in tqdm(val_iterator):
                        try:
                            data = next(val_iterator)
                        except StopIteration:
                            # Reset the iterator when exhausted
                            val_iterator = iter(val_dataloader)
                            data = next(val_iterator)
                        torch.cuda.empty_cache()
                        # data = next(val_iterator)
                        pixel_values = data["pixel_values"]
                        labels = data["labels"].to(self.vla.device)
                        attention_mask = data["attention_mask"].to(self.vla.device)
                        input_ids = data["input_ids"].to(self.vla.device)

                        if len(maskidx)==1 and maskidx[0]==6:
                            # fiter wrong predict samples / check the success rate of ori model
                            pre_ids = self.randomPatchTransform.im_process(pixel_values, mean=self.mean, std=self.std)
                            pre_output: CausalLMOutputWithPast = self.vla(
                                input_ids=input_ids.to(self.vla.device),
                                attention_mask=attention_mask.to(self.vla.device),
                                pixel_values=pre_ids.to(torch.bfloat16).to(self.vla.device),
                                labels=labels,
                            )
                            pre_action_logits = pre_output.logits[:,
                                            self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
                            pre_action_preds = pre_action_logits.argmax(dim=2)
                            pre_action_gt = labels[:, 1:].to(self.vla.device)
                            pre_mask = pre_action_gt > self.action_tokenizer.action_token_begin_idx

                            formulate_pre_pred = pre_action_preds[pre_mask].view(pre_action_preds[pre_mask].shape[0] // 7,7)  # shape [2,7]
                            formulate_pre_gt = pre_action_gt[pre_mask].view(pre_action_gt[pre_mask].shape[0] // 7,7)
                            correct_index = []
                            for del_idx in range(formulate_pre_pred.shape[0]):
                                if formulate_pre_pred[del_idx][-1] == formulate_pre_gt[del_idx][-1]:
                                    correct_index.append(del_idx)
                            # modifiy the input
                            if len(correct_index) == 0:
                                print("No Correct in Val!")
                                continue
                            else:
                                labels = labels[correct_index,:]
                                attention_mask = attention_mask[correct_index,:]
                                input_ids = input_ids[correct_index,:]
                                pixel_values = [pixel_values[i] for i in correct_index]
                        val_num_sample += labels.shape[0]
                        # process img
                        # modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch, mean=self.mean, std=self.std)
                        # modified_images = self.randomPatchTransform.random_paste_patch(pixel_values, patch, mean=self.mean,std=self.std)
                        if not geometry and not colorjitter:
                            modified_images = self.randomPatchTransform.paste_patch_fix(pixel_values, patch, mean=self.mean,
                                                                                        std=self.std)
                        else:
                            modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch,
                                                                                                 mean=self.mean,
                                                                                                 std=self.std,
                                                                                                 geometry=geometry,
                                                                                                 colorjitter=colorjitter)
                        newlabels = []
                        for k in range(labels.shape[0]):
                            temp_label = labels[k].clone()
                            temp_label[temp_label != -100] = target_action
                            newlabels.append(temp_label.unsqueeze(0))
                        newlabels = torch.cat(newlabels, dim=0)
                        output: CausalLMOutputWithPast = self.vla(
                            input_ids=input_ids.to(self.vla.device),
                            attention_mask=attention_mask.to(self.vla.device),
                            pixel_values=modified_images.to(torch.bfloat16).to(self.vla.device),
                            labels=newlabels,
                        )
                        action_logits = output.logits[:, self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
                        action_preds = action_logits.argmax(dim=2)
                        action_gt = newlabels[:, 1:].to(action_preds.device)
                        mask = action_gt > self.action_tokenizer.action_token_begin_idx
                        continuous_actions_pred = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                        )
                        continuous_actions_gt = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                        )
                        val_inner_relatived_distance += self.calculate_relative_distance_target(continuous_actions_pred,
                                                                                        continuous_actions_gt)
                        real_gt = labels[:, 1:].to(action_preds.device)
                        real_mask = real_gt > self.action_tokenizer.action_token_begin_idx
                        real_continuous_actions_gt = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(real_gt[real_mask].cpu().numpy())
                        )
                        if len(maskidx)==1 and maskidx[0]==6:
                            temp_02other_success, gt_0_num, temp_12other_success, gt_1_num, temp_other20_success, gt_others_num = self.calculate_01_ASR(pred=action_preds[mask],gt=labels[:,1:][mask])
                            all_02other_success += temp_02other_success
                            all_gt_0_num += gt_0_num
                            all_12other_success += temp_12other_success
                            all_gt_1_num += gt_1_num
                            all_other20_success += temp_other20_success
                            all_gt_others_num += gt_others_num
                        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                        # print(f"continuous_actions_pred: {continuous_actions_pred}, {continuous_actions_pred.shape}")
                        # print(f"continuous_actions_gt: {continuous_actions_gt}", {continuous_actions_gt.shape})
                        temp_continuous_actions_pred = continuous_actions_pred.view(continuous_actions_pred.shape[0] // len(maskidx),len(maskidx)) # shape [2,7]
                        temp_continuous_actions_gt = continuous_actions_gt.view(continuous_actions_gt.shape[0] // len(maskidx),len(maskidx))
                        # print(f"temp_continuous_actions_pred: {temp_continuous_actions_pred}, {temp_continuous_actions_pred.shape}")
                        # print(f"temp_continuous_actions_gt: {temp_continuous_actions_gt}, {temp_continuous_actions_gt.shape}")
                        # if continuous_actions_gt==1:
                        temp_real_continuous_actions_gt = real_continuous_actions_gt.view(
                            real_continuous_actions_gt.shape[0] // 7, 7)  # shape [2,7]
                        cated = torch.cat(
                            [temp_real_continuous_actions_gt[:, item].unsqueeze(1) for item in maskidx], dim=1)
                        if len(maskidx)==1 and maskidx[0]==6:
                            # [round(float(item), 2) for item in real_continuous_actions_gt.tolist()]
                            # print(f"continuous_actions_pred: {continuous_actions_pred},continuous_actions_gt: {continuous_actions_gt},real_continuous_actions_gt: {real_continuous_actions_gt}")
                            print("--------UP-------")
                            # print(f"continuous_actions_pred: {[round(float(item), 2) for item in continuous_actions_pred.tolist()]},\n continuous_actions_gt: {[round(float(item), 2) for item in continuous_actions_gt.tolist()]},\n real_continuous_actions_gt: {[round(float(item), 2) for item in real_continuous_actions_gt.tolist()]}")
                            print(f"continuous_actions_pred: {[round(float(item), 2) for item in continuous_actions_pred.tolist()]},\n continuous_actions_gt: {[round(float(item), 2) for item in continuous_actions_gt.tolist()]},\n real_continuous_actions_gt: {cated}")
                            print("---------DOWN------")
                        else:
                            print("--------UP-------")
                            # temp_real_continuous_actions_gt = real_continuous_actions_gt.view(
                            #     real_continuous_actions_gt.shape[0] // 7, 7)  # shape [2,7]
                            # cated = torch.cat(
                            #     [temp_real_continuous_actions_gt[:, item].unsqueeze(1) for item in maskidx], dim=1)
                            print(f"continuous_actions_pred: {continuous_actions_pred},continuous_actions_gt: {cated}")
                            print("---------DOWN------")
                        # success_attack_num += (torch.sum(continuous_actions_pred-continuous_actions_gt,dim=0) == 0).sum().item()
                        for idx in range(temp_continuous_actions_pred.shape[0]):
                            if temp_continuous_actions_pred.ndim==2:
                                # multilabel
                                flag = True
                                for idy in range(temp_continuous_actions_pred.shape[1]):
                                    if temp_continuous_actions_pred[idx,idy]!=temp_continuous_actions_gt[idx,idy]:
                                        flag = False
                                if flag:
                                    success_attack_num += 1
                            else:
                                # single label
                                if continuous_actions_pred[idx]==continuous_actions_gt[idx]:
                                    success_attack_num+=1
                        # Cal DoE ASR
                        # temp_doe_ASR = self.calculate_ASR_by_DoE(temp_continuous_actions_pred,temp_continuous_actions_gt)
                        avg_L1_loss += action_l1_loss.item()
                        avg_CE_loss += output.loss.item()
                    avg_L1_loss /= val_num_sample
                    avg_CE_loss /= val_num_sample
                    ASR = success_attack_num / val_num_sample
                    val_inner_relatived_distance /= val_num_sample
                        # all_02other_success, all_gt_0_num, all_12other_success, all_gt_1_num = 0, 0, 0, 0
                    if len(maskidx)==1 and maskidx[0]==6:
                        # ASR_02other = all_02other_success / all_gt_0_num
                        # ASR_12other = all_12other_success / all_gt_1_num
                        # ALL_ASR_6 = (all_02other_success+all_12other_success)/(all_gt_0_num+all_gt_0_num)
                        # 防止除以零
                        ASR_02other = all_02other_success / all_gt_0_num if all_gt_0_num != 0 else 0
                        ASR_12other = all_12other_success / all_gt_1_num if all_gt_1_num != 0 else 0
                        ASR_other20 = all_other20_success / all_gt_others_num if all_gt_others_num != 0 else 0
                        ALL_ASR_6 = (all_02other_success + all_12other_success) / (all_gt_0_num + all_gt_1_num) if (all_gt_0_num + all_gt_1_num) != 0 else 0
                        # wandb.log({"bar_chart": bar_chart})
                        print(f"all_02other_success: {all_02other_success}, all_gt_0_num: {all_gt_0_num}\n all_12other_success: {all_12other_success}, all_gt_1_num: {all_gt_1_num}\n all_other20_success: {all_other20_success}, all_gt_others_num: {all_gt_others_num}")
                        print(f"success_attack_num: {success_attack_num}, val_num_sample: {val_num_sample}, ASR: {ASR}")
                        wandb.log(
                            {
                                "VAL_avg_CE_loss": avg_CE_loss,
                                "VAL_avg_L1_loss": avg_L1_loss,
                                "VAL_ASR(pred0-AllCorrect)": ASR,
                                "ASR_02other": ASR_02other,
                                "ASR_12other": ASR_12other,
                                "ASR_other20": ASR_other20,
                                "ALL_ASR_6":ALL_ASR_6,
                                "inner_relatived_distance":inner_relatived_distance
                            },
                            step=i,
                        )
                    else:
                        # wandb.log({"bar_chart": bar_chart})
                        print(f"success_attack_num: {success_attack_num}, val_num_sample: {val_num_sample}, ASR: {ASR}")
                        wandb.log(
                            {
                                "VAL_avg_CE_loss": avg_CE_loss,
                                "VAL_avg_L1_loss": avg_L1_loss,
                                "VAL_ASR": ASR,
                                "VAL_inner_relatived_distance":val_inner_relatived_distance
                            },
                            step=i,
                        )

                    # save
                    if avg_L1_loss<self.min_val_avg_L1_loss:
                        self.min_val_avg_L1_loss = avg_L1_loss
                        temp_save_dir = os.path.join(self.save_dir,f"{str(i)}")
                        os.makedirs(temp_save_dir,exist_ok=True)
                        torch.save(patch.detach().cpu(), os.path.join(temp_save_dir,"patch.pt"))
                        val_related_file_path = os.path.join(temp_save_dir,"val_related_data")
                        os.makedirs(val_related_file_path,exist_ok=True)
                        torch.save(continuous_actions_pred.detach().cpu(), os.path.join(val_related_file_path,"continuous_actions_pred.pt"))
                        torch.save(continuous_actions_gt.detach().cpu(), os.path.join(val_related_file_path,"continuous_actions_gt.pt"))
                        modified_images = self.randomPatchTransform.denormalize(modified_images[:,0:3,:,:].detach().cpu(), mean=self.mean[0], std=self.std[0])
                        pil_imgs = []
                        for o in range(modified_images.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(modified_images[o,:,:,:])
                            pil_img.save(os.path.join(val_related_file_path,f"{str(o)}.png"))
                            pil_imgs.append(pil_img)
                        wandb.log({"AdvImg": [wandb.Image(pil_img) for pil_img in pil_imgs]})
                    # save last checkpoint
                    temp_save_dir = os.path.join(self.save_dir, "last")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    torch.save(continuous_actions_pred.detach().cpu(),
                               os.path.join(val_related_file_path, "continuous_actions_pred.pt"))
                    torch.save(continuous_actions_gt.detach().cpu(),
                               os.path.join(val_related_file_path, "continuous_actions_gt.pt"))
                    modified_images = self.randomPatchTransform.denormalize(
                        modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                self.val_CE_loss.append(avg_CE_loss)
                self.val_L1_loss.append(avg_L1_loss)
                self.val_ASR.append(success_attack_num/val_num_sample)
                self.val_inner_relatived_distance.append(val_inner_relatived_distance)
                # save here
                self.save_info(path=self.save_dir)
                torch.cuda.empty_cache()        # torch.save(modified_images.detach().cpu(), os.path.join(val_related_file_path,"modified_images.pt"))

    def modifiy_labels(self,labels,target_action={"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8}):
        newlabels = []
        for j in range(labels.shape[0]):
            temp_label = labels[j]
            # temp_label[temp_label != -100] = target_action
            first_valid_index = (temp_label != -100).nonzero(as_tuple=True)[0].item()
            for key,value in target_action.items():
                if value != -100:
                    temp_label[int(first_valid_index+int(key))] = value
            newlabels.append(temp_label.unsqueeze(0))
            print(temp_label)
        newlabels = torch.cat(newlabels, dim=0)
        return newlabels

    def calculate_ASR_by_DoE(self,pred,gt):
        assert pred.shape == gt.shape
        temp_doe_ASR = [0,0,0,0,0,0,0]
        for pred_idx in range(pred.shape[0]):
            for DoE_idx in range(pred.shape[1]):
                if pred[pred_idx,DoE_idx]==gt[pred_idx,DoE_idx]:
                    temp_doe_ASR[DoE_idx] += 1
        return temp_doe_ASR

    def calculate_01_ASR(self,pred,gt):
        temp_02other_success = 0
        gt_0_num = 0
        temp_12other_success = 0
        gt_1_num = 0
        gt_others_num = 0
        temp_other20_success = 0
        print(f"calculate_01_ASR:{gt}")
        for idx in range(gt.shape[0]):
            if gt[idx] == 31872: # gt is 0
                gt_0_num += 1
                if pred[idx] != 31872:
                    temp_02other_success += 1
            elif gt[idx] == 31744: # gt is 1
                gt_1_num += 1
                if pred[idx] != 31744:
                    temp_12other_success += 1
            elif gt[idx] != 31872: # gt is not 0
                gt_others_num+=1
                if pred[idx] == 31872:
                    temp_other20_success += 1

        return temp_02other_success, gt_0_num, temp_12other_success, gt_1_num, temp_other20_success, gt_others_num

    def filter_train(self,data):
        pixel_values = data["pixel_values"]
        labels = data["labels"].to(self.vla.device)
        attention_mask = data["attention_mask"].to(self.vla.device)
        input_ids = data["input_ids"].to(self.vla.device)

        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        one_index = []
        for idx in range(masked_labels.shape[0]):
            if masked_labels[idx,6]==31744:
                one_index.append(idx)
        if 1<len(one_index)<8:
            labels = labels[one_index, :]
            attention_mask = attention_mask[one_index, :]
            input_ids = input_ids[one_index, :]
            pixel_values = [pixel_values[i] for i in one_index]
        elif len(one_index)>8:
            chosen = random.sample(one_index,k=8)
            labels = labels[chosen, :]
            attention_mask = attention_mask[chosen, :]
            input_ids = input_ids[chosen, :]
            pixel_values = [pixel_values[i] for i in chosen]
        elif one_index is None:
            chosen = random.sample(range(labels.shape[0]),k=8)
            labels = labels[chosen, :]
            attention_mask = attention_mask[chosen, :]
            input_ids = input_ids[chosen, :]
            pixel_values = [pixel_values[i] for i in chosen]
        # print(f"one_index: {one_index}, labels.shape: {labels.shape}")
        return labels, attention_mask, input_ids, pixel_values

    def save_info(self,path):
        with open(os.path.join(path,'val_CE_loss.pkl'), 'wb') as file:
            pickle.dump(self.val_CE_loss, file)
        with open(os.path.join(path,'val_L1_loss.pkl'), 'wb') as file:
            pickle.dump(self.val_L1_loss, file)
        with open(os.path.join(path,'val_ASR.pkl'), 'wb') as file:
            pickle.dump(self.val_ASR, file)
        with open(os.path.join(path,'val_inner_relatived_distance.pkl'), 'wb') as file:
            pickle.dump(self.val_inner_relatived_distance, file)
        with open(os.path.join(path,'train_CE_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_CE_loss, file)
        with open(os.path.join(path,'train_inner_avg_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_inner_avg_loss, file)
        with open(os.path.join(path,'train_inner_relatived_distance.pkl'), 'wb') as file:
            pickle.dump(self.train_inner_relatived_distance, file)

    def calculate_relative_distance_target(self, pred,gt):
        # process input -> (n,) -> (a,b)
        # pred = pred.clone().view(pred.shape[0]//7,7)
        # gt = gt.clone().view(gt.shape[0]//7,7)
        # # filter
        # all_pred = []
        # all_gt = []
        # for item in maskidx:
        #     all_pred.append(pred[:,item].unsqueeze(0)) # [action, bs]
        #     all_gt.append(gt[:,item].unsqueeze(0)) # [action, bs]
        # pred = torch.cat(all_pred, dim=0) # [action, bs]
        # gt = torch.cat(all_gt, dim=0) # [action, bs]
        relative_distance = 0
        for idx1 in range(pred.shape[0]): # go through each action
            anchor = gt[idx1]
            input_point = pred[idx1]
            # define the upper and lower bound of the input range
            upper_bound = 1
            lower_bound = -1
            # cal the distance to upper and lower bound for anchor
            distance_to_upper = upper_bound - anchor
            distance_to_lower = anchor - lower_bound
            max_boundary_distance = torch.max(distance_to_upper, distance_to_lower)
            distance_to_anchor = torch.abs(input_point - anchor)
            # cal relative distance
            temp_relative_distance = distance_to_anchor / max_boundary_distance
            relative_distance += temp_relative_distance
        return relative_distance/pred.shape[0]