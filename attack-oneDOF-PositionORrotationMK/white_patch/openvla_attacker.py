# image processor
# resize-naive - 224 224 interpolations - bicubic
import torch
import torchvision
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
IGNORE_INDEX = -100

def normalize(images,mean,std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images,mean,std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

# de_im_tensor1 = denormalize(im_tensor1,torch.tensor([ 0.484375,  0.455078125,  0.40625]).to(im_tensor1.device),torch.tensor([0.228515625,0.2236328125, 0.224609375]).to(im_tensor1.device))
# de_im_tensor2 = denormalize(im_tensor2,torch.tensor([0.5,0.5,0.5]).to(im_tensor2.device),torch.tensor([0.5,0.5,0.5]).to(im_tensor2.device))

# if __name__ == '__main__':
#     import torch
#     from transformers import AutoConfig
#     from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
#     from transformers import AutoModelForVision2Seq, AutoProcessor
#     from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
#     from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
#     from datetime import datetime
#     import os
#     from PIL import Image
#     import numpy as np
#     import sys
#
#     sys.path.append("/spl_data/tw9146/openvla-main/attack/white")
#     from openvla_attacker import OpenVLAAttacker
#
#     current_time = datetime.now()
#     year = current_time.year
#     month = current_time.month
#     day = current_time.day
#     hour = current_time.hour
#     minute = current_time.minute
#     path = f"/spl_data/tw9146/openvla-main/run/white_attack/{year}_{month}_{day}-{hour}-{minute}"
#
#     AutoConfig.register("openvla", OpenVLAConfig)
#     AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
#     AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
#     quantization_config = None
#     processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
#
#     os.makedirs(path, exist_ok=True)
#     openVLA_Attacker = OpenVLAAttacker("vla_placeholder", processor, path)
#     img = Image.open("/spl_data/tw9146/openvla-main/0dataprocess/test.png")
#     prompt = "put down blue can"
#     openVLA_Attacker.attack_unconstrained(prompt, img, num_iter=5000, action=np.zeros(7), alpha=1 / 255)
#     print("Attack done!")
class OpenVLAAttacker(object):
    def __init__(self, vla, processor, save_dir="",optimizer="pgd"):
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
        self.randomPatchTransform = RandomPatchTransform(self.vla.device)
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



    def patchattack_unconstrained(self,train_dataloader,val_dataloader,num_iter=5000,target_action=np.zeros(7),patch_size=[3,50,50],alpha=1/255, accumulate_steps=1):
        patch = torch.randn(patch_size).to(self.vla.device)
        patch.requires_grad_(True)
        patch.retain_grad()
        target_action = self.base_tokenizer(self.action_tokenizer(target_action)).input_ids[2:]
        target_action.append(2)
        target_action = list(target_action)
        target_action = torch.tensor(target_action).to(self.vla.device)
        if self.optimizer == "adamW":
            optimizer = torch.optim.AdamW([patch], lr=alpha)
            # ....
            import transformer


        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader)
        # for idx, data in enumerate(train_dataloader):
        for i in tqdm(range(num_iter)):
            torch.cuda.empty_cache()
            data = next(train_iterator)
            pixel_values = data["pixel_values"]
            labels = data["labels"].to(self.vla.device)
            attention_mask = data["attention_mask"].to(self.vla.device)
            input_ids = data["input_ids"].to(self.vla.device)

            # process img
            modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch, mean=self.mean, std=self.std)
            # modified_images = self.randomPatchTransform.random_paste_patch(pixel_values, patch, mean=self.mean, std=self.std)
            # process label
            newlabels = []
            for j in range(labels.shape[0]):
                temp_label = labels[j]
                temp_label[temp_label != -100] = target_action
                newlabels.append(temp_label.unsqueeze(0))
            newlabels = torch.cat(newlabels, dim=0)

            output: CausalLMOutputWithPast = self.vla(
                input_ids=input_ids.to(self.vla.device),
                attention_mask=attention_mask.to(self.vla.device),
                pixel_values=modified_images.to(torch.bfloat16).to(self.vla.device),
                labels=newlabels,
            )
            loss = output.loss
            loss.backward()
            log_patch_grad = patch.grad.detach().mean().item()
            if self.optimizer == "adamW":

                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                    optimizer.step()
                    patch.data = patch.data.clamp(0, 1)
                    optimizer.zero_grad()
                    self.vla.zero_grad()

            elif self.optimizer == "pgd":
                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                    patch.data = (patch.data - alpha * patch.grad.detach().sign()).clamp(0, 1)
                    self.vla.zero_grad()
                    patch.grad.zero_()

            # patch.grad.zero_()
            # self.vla.zero_grad()
            self.loss_buffer.append(loss.item())
            print("target_loss: %f" % (loss.item()))
            wandb.log(
                {
                    "TRAIN_attack_loss(CE)": loss.item(),
                    "TRAIN_patch_gradient": log_patch_grad,
                },
                step=i,
            )
            if i % 100 == 0:
                self.plot_loss()

            if i % 200 == 0:
                avg_CE_loss = 0
                avg_L1_loss = 0
                val_num_sample = 0
                success_attack_num = 0
                print("evaluating...")
                with torch.no_grad():
                    for j in tqdm(range(100)):
                        torch.cuda.empty_cache()
                        data = next(val_iterator)
                        pixel_values = data["pixel_values"]
                        labels = data["labels"].to(self.vla.device)
                        attention_mask = data["attention_mask"].to(self.vla.device)
                        input_ids = data["input_ids"].to(self.vla.device)
                        val_num_sample += labels.shape[0]

                        # process img
                        modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch, mean=self.mean, std=self.std)
                        # modified_images = self.randomPatchTransform.random_paste_patch(pixel_values, patch, mean=self.mean,std=self.std)
                        newlabels = []
                        for k in range(labels.shape[0]):
                            temp_label = labels[k]
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
                        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                        # print(f"continuous_actions_pred: {continuous_actions_pred}, {continuous_actions_pred.shape}")
                        # print(f"continuous_actions_gt: {continuous_actions_gt}", {continuous_actions_gt.shape})
                        temp_continuous_actions_pred = continuous_actions_pred.view(continuous_actions_pred.shape[0] // 7,7)
                        temp_continuous_actions_gt = continuous_actions_gt.view(continuous_actions_gt.shape[0] // 7,7)
                        success_attack_num += (torch.sum(temp_continuous_actions_pred-temp_continuous_actions_gt,dim=1) == 0).sum().item()
                        avg_L1_loss += action_l1_loss.item()
                        avg_CE_loss += output.loss.item()
                    avg_L1_loss /= val_num_sample
                    avg_CE_loss /= val_num_sample
                    ASR = success_attack_num / val_num_sample
                    wandb.log(
                        {
                            "VAL_avg_CE_loss": avg_CE_loss,
                            "VAL_avg_L1_loss": avg_L1_loss,
                            "VAL_ASR": ASR,
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