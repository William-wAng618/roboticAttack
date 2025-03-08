import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb
import torch.nn.functional as F
from white_patch.appply_random_transform import RandomPatchTransform
from white_patch.openvla_dataloader import get_dataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
import pickle
from tqdm import tqdm
import os
import transformers
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
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
    def __init__(self, vla_path, dataset_name, save_dir="", resize_patch=False,patch_size=[3,50,50],lr=0.01,bs=1,warmup=20,num_iter=10000,maskidx=[],innerLoop=1,geometry=True,use_wandb=True, MSE_weights=1):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        quantization_config = None
        self.processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        for param in self.vla.parameters():
            param.requires_grad_(False)
        self.train_dataset, self.val_dataset = get_dataset(dataset=dataset_name)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.save_dir = save_dir
        self.randomPatchTransform = RandomPatchTransform(self.vla.device,resize_patch)
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.MSE_Distance_best = 1000000
        self.collator = PaddedCollatorForActionPrediction(
            self.processor.tokenizer.model_max_length, self.processor.tokenizer.pad_token_id, padding_side="right"
        )
        self.bs = bs
        self.lr = lr
        self.warmup = warmup
        self.num_iter = num_iter
        self.maskidx = maskidx
        self.innerLoop = innerLoop
        self.geometry = geometry
        self.use_wandb = use_wandb
        self.patch_size = patch_size
        self.val_CE_loss = []
        self.val_MSE_Distance = []
        self.val_UAD = []
        self.MSE_weights = MSE_weights


    # def setup(self, rank, world_size):
    #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    #     torch.cuda.set_device(rank)

    def setup(self, rank, world_size):
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def cleanup(self):
        dist.destroy_process_group()

    def mask_labels(self,labels,maskidx):
        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        template_labels = torch.ones_like(masked_labels,device=masked_labels.device)*-100
        for idx in maskidx:
            template_labels[:, idx] = masked_labels[:, idx]
        labels[labels > 2] = template_labels.view(-1)
        return labels

    def weighted_loss(self, logits, labels, rank, MSE_weights):
        temp_label = labels[:,1:].to(rank) # (bs,seq_len) remove bos token
        action_mask = temp_label > 2
        temp_logits = logits[:,:,31744:32000] # (bs,seq_len,256) only consider the 256 classes for the target class
        action_logits = temp_logits[:,-temp_label.shape[-1]-1:-1, :] # shift logits see modeling_llama.py line 1233
        action_logits = action_logits[action_mask]
        reweigh = torch.arange(1,257).to(logits.device)/256 # [1,,...256]
        temp_prob = F.softmax(action_logits,dim=-1) # [bs,action_length,256]
        reweighted_prob = (temp_prob * reweigh).sum(dim=-1)  # [bs, action_length]
        hard_max_labels = temp_label[action_mask]
        hard_max_labels[hard_max_labels > 31872]=31999
        hard_max_labels[hard_max_labels <= 31872]=31744
        hard_max_labels[hard_max_labels == 31999]=1/256
        hard_max_labels[hard_max_labels == 31744]=1
        UAD = self.cal_UAD(action_logits.argmax(dim=-1)+31744,temp_label[action_mask])
        distance_loss = F.mse_loss(MSE_weights*reweighted_prob.contiguous(), MSE_weights*hard_max_labels.float().contiguous())

        # targeted max distance CE loss
        # ce_action_logits = logits[:,-temp_label.shape[-1]-1:-1, :]
        # ce_action_logits = ce_action_logits[action_mask]
        # ce_hard_max_labels = temp_label[action_mask]
        # ce_hard_max_labels[ce_hard_max_labels > 31872]=31744
        # ce_hard_max_labels[ce_hard_max_labels <= 31872]=31999
        # ce_loss = F.cross_entropy(ce_action_logits, ce_hard_max_labels)
        # distance_loss = distance_loss + ce_loss
        return distance_loss, UAD

    def cal_UAD(self,pred,gt):
        continuous_actions_gt = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(gt.clone().detach().cpu().numpy()), device=gt.device
        )
        continuous_actions_pred = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(pred.clone().detach().cpu().numpy()), device=pred.device
        )
        max_distance = torch.where(continuous_actions_gt > 0, torch.abs(continuous_actions_gt - (-1)), torch.abs(continuous_actions_gt - 1))
        distance = torch.abs(continuous_actions_pred - continuous_actions_gt)
        UAD = (distance / max_distance).mean()
        return UAD

    def attack(self, rank, world_size):
        self.setup(rank, world_size)
        if int(os.environ.get("RANK", 0)) == 0:
            patch = torch.rand(self.patch_size).to('cuda')
        else:
            patch = torch.empty(self.patch_size).to('cuda')
        dist.broadcast(patch, src=0)
        patch = torch.nn.Parameter(patch)
        self.vla.eval()
        self.vla.patch = patch
        self.vla.register_parameter("patch", self.vla.patch)
        self.vla.patch.requires_grad_(True)
        self.vla.patch.retain_grad()

        # train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)
        # train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs,collate_fn=self.collator, sampler=train_sampler) # num_worker?
        # val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank)
        # val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.bs,collate_fn=self.collator, sampler=val_sampler) # num_worker?

        self.train_dataset = self.train_dataset.shard(num_shards=world_size, index=rank)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs, collate_fn=self.collator)
        self.val_dataset = self.val_dataset.shard(num_shards=world_size, index=rank)
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.bs, collate_fn=self.collator)

        # train_iterator = iter(train_dataloader)
        # val_iterator = iter(val_dataloader)

        model = self.vla.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optimizer = transformers.AdamW([model.module.patch], lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=int(self.num_iter),
            num_cycles=0.5,
            last_epoch=-1,
        )
        # for i in tqdm(range(self.num_iter)):
        with tqdm(total=self.num_iter, desc="Training") as pbar:
            for i, data in enumerate(train_dataloader):
                if i == self.num_iter:
                    break
                # data = next(train_iterator)
                pixel_values = data["pixel_values"]
                labels = data["labels"].to(rank)
                attention_mask = data["attention_mask"].to(rank)
                input_ids = data["input_ids"].to(rank)

                # print("masking labels...")
                labels = self.mask_labels(labels, self.maskidx)

                for inner_loop in range(self.innerLoop):
                    optimizer.zero_grad()
                    # if self.geometry:
                    modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, model.module.patch,
                                                                                         mean=self.mean,
                                                                                         std=self.std,
                                                                                         geometry=self.geometry)
                    output: CausalLMOutputWithPast = model(
                        input_ids=input_ids.to(rank),
                        attention_mask=attention_mask.to(rank),
                        pixel_values=modified_images.to(torch.bfloat16).to(rank),
                        labels=labels,
                    )
                    celoss = output.loss
                    MSE_Distance, UAD = self.weighted_loss(output.logits, labels, rank, self.MSE_weights)
                    # MSE_Distance = MSE_Distance # + 1/celoss
                    # MSE_Distance = MSE_Distance  + 1/celoss
                    MSE_Distance.backward()
                    log_patch_grad = model.module.patch.grad.detach().mean().item()
                    optimizer.step()
                    model.module.patch.data = model.module.patch.data.clamp(0, 1)
                    # optimizer.zero_grad()
                    # model.zero_grad()
                    # torch.cuda.empty_cache()
                scheduler.step()
                gobal_train_CE_loss = torch.tensor([celoss], dtype=torch.float32, device=rank)
                dist.all_reduce(gobal_train_CE_loss, op=dist.ReduceOp.AVG)
                gobal_log_patch_grad = torch.tensor([log_patch_grad], dtype=torch.float32, device=rank)
                dist.all_reduce(gobal_log_patch_grad, op=dist.ReduceOp.MAX)
                gobal_MSE_Distance = torch.tensor([MSE_Distance], dtype=torch.float32, device=rank)
                dist.all_reduce(gobal_MSE_Distance, op=dist.ReduceOp.AVG)
                gobal_UAD = torch.tensor([UAD], dtype=torch.float32, device=rank)
                dist.all_reduce(gobal_UAD, op=dist.ReduceOp.AVG)

                train_logdata = {"TRAIN_attack_loss(CE)": gobal_train_CE_loss,
                                 "TRAIN_patch_gradient": gobal_log_patch_grad,
                                 "TRAIN_LR": optimizer.param_groups[0]["lr"],
                                 "TRAIN_attack_loss (MSE_Distance)": gobal_MSE_Distance,
                                 "TRAIN_UAD": gobal_UAD}
                pbar.update(1)
                if rank == 0:
                    if self.use_wandb:
                        wandb.log(train_logdata, step=i)

                if i % 200 == 0:
                    avg_CE_loss = torch.tensor(0, dtype=torch.float32)
                    val_num_sample = torch.tensor(0, dtype=torch.float32)
                    avg_MSE_Distance = torch.tensor(0, dtype=torch.float32)
                    avg_UAD = torch.tensor(0, dtype=torch.float32)
                    torch.cuda.empty_cache()

                    with torch.no_grad():
                        val_iter = 100
                        with tqdm(total=val_iter, desc="Validating") as pbar_val:
                            for j, data in enumerate(val_dataloader):
                                if j == val_iter:
                                    break
                                pixel_values = data["pixel_values"]
                                labels = data["labels"].to(rank)
                                attention_mask = data["attention_mask"].to(rank)
                                input_ids = data["input_ids"].to(rank)
                                val_num_sample += labels.shape[0]
                                # if self.geometry:
                                modified_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values,
                                                                                                     model.module.patch,
                                                                                                     mean=self.mean,
                                                                                                     std=self.std,
                                                                                                     geometry=self.geometry)
                                labels = self.mask_labels(labels, self.maskidx)
                                output: CausalLMOutputWithPast = model(
                                    input_ids=input_ids.to(rank),
                                    attention_mask=attention_mask.to(rank),
                                    pixel_values=modified_images.to(torch.bfloat16).to(rank),
                                    labels=labels,
                                )
                                val_MSE_Distance, val_UAD = self.weighted_loss(output.logits, labels, rank, self.MSE_weights)

                                avg_MSE_Distance += val_MSE_Distance.item()
                                avg_UAD += val_UAD.item()
                                avg_CE_loss += output.loss.item()
                                pbar_val.update(1)
                            torch.cuda.empty_cache()
                            avg_MSE_Distance /= val_iter
                            avg_UAD /= val_iter
                            avg_CE_loss/= val_iter
                            # collect
                            gobal_avg_MSE_Distance = torch.tensor([avg_MSE_Distance], dtype=torch.float32, device=rank)
                            gobal_avg_UAD = torch.tensor([avg_UAD], dtype=torch.float32, device=rank)
                            gobal_avg_CE_loss = torch.tensor([avg_CE_loss], dtype=torch.float32, device=rank)
                            dist.all_reduce(gobal_avg_MSE_Distance, op=dist.ReduceOp.AVG)
                            dist.all_reduce(gobal_avg_UAD, op=dist.ReduceOp.AVG)
                            dist.all_reduce(gobal_avg_CE_loss, op=dist.ReduceOp.AVG)
                            log_data = {}
                            log_data["VAL_MSE_Distance"] = gobal_avg_MSE_Distance
                            log_data["VAL_UAD"] = gobal_avg_UAD


                            if rank == 0:
                                if gobal_avg_MSE_Distance < self.MSE_Distance_best:
                                    self.MSE_Distance_best = gobal_avg_MSE_Distance
                                    temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                                    os.makedirs(temp_save_dir, exist_ok=True)
                                    torch.save(model.module.patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                                    os.makedirs(val_related_file_path, exist_ok=True)
                                    modified_images = self.randomPatchTransform.denormalize(
                                        modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                                    pil_imgs = []
                                    for o in range(modified_images.shape[0]):
                                        pil_img = torchvision.transforms.ToPILImage()(modified_images[o, :, :, :])
                                        pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                                        pil_imgs.append(pil_img)
                                    if self.use_wandb != "false":
                                        wandb.log(log_data, step=i)
                                        wandb.log({"AdvImg": [wandb.Image(pil_img) for pil_img in pil_imgs]})
                                temp_save_dir = os.path.join(self.save_dir, "last")
                                os.makedirs(temp_save_dir, exist_ok=True)
                                torch.save(model.module.patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                                val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                                os.makedirs(val_related_file_path, exist_ok=True)
                                modified_images = self.randomPatchTransform.denormalize(
                                    modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                                # # upload last image to wandb evey 1000 steps, avoid wandb timeout
                                # if i % 200 == 0:
                                #     pil_imgs = []
                                #     for o in range(modified_images.shape[0]):
                                #         pil_img = torchvision.transforms.ToPILImage()(modified_images[o, :, :, :])
                                #         pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                                #         pil_imgs.append(pil_img)
                                #     if self.use_wandb != "false":
                                #         wandb.log({"Last_Step_AdvImg": [wandb.Image(pil_img) for pil_img in pil_imgs]})
                            if rank==0:
                                self.val_CE_loss.append(gobal_avg_CE_loss)
                                self.val_MSE_Distance.append(gobal_avg_MSE_Distance)
                                self.val_UAD.append(gobal_avg_UAD)
                            torch.cuda.empty_cache()
        self.cleanup()

    @classmethod
    def run(cls, vla_path,dataset_name, save_dir, resize_patch, patch_size,lr,bs,warmup,num_iter,maskidx,innerLoop,geometry,use_wandb, MSE_weights):
        world_size = torch.cuda.device_count()
        instance_params = {
            "vla_path": vla_path, "dataset_name":dataset_name, "save_dir": save_dir, "resize_patch": resize_patch, "patch_size": patch_size,
            "lr": lr, "bs": bs, "warmup": warmup,
            "num_iter": num_iter, "maskidx": maskidx, "innerLoop": innerLoop, "geometry": geometry,
            "use_wandb": use_wandb, "MSE_weights": MSE_weights
        }
        mp.spawn(OpenVLAAttacker._attack_entry, args=(instance_params, world_size), nprocs=world_size)

    @staticmethod
    def _attack_entry(rank, instance_params, world_size):
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

        instance = OpenVLAAttacker(**instance_params)
        instance.attack(rank, world_size)
