# Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics



[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

<div align="center">
  <img src=".\fig\mainfig.png">
</div>
<p>
Overall Adversarial Framework. 
</p>

Built on top of [OpenVLA](https://github.com/openvla/openvla), a remarkable generalist vision-language-action model work. 

---

## Latest Updates
- [2025-03-08]
  - Upate UADA.
  - Bug fix.
  - Simple and flexible new evaluation tool (See in roboticAttack/evaluation_tool)
  - Release new ver. of Paper (ARXIV Release eta 11 Mar 2025 00:00:00 GMT).
  - ALL patches released (See in roboticAttack
/adversarial_patches).
- [2024-11-26] Pre release

---
## Important Resources
- Adversarial Patches (See adversarial_patches)
- Video (See videos)
---

## 1.Installation
(a) Use the setup commands below to get started:

```bash
conda create -n roboticAttack python=3.10 -y
conda activate roboticAttack

# Install PyTorch.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Clone and install this repo
git clone https://github.com/William-wAng618/roboticAttack.git
cd roboticAttack
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

(b) Install LIBERO evaluation environment

```bash
# install LIBERO repo
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
# install other required packages:
cd ../ # cd roboticAttack/
pip install -r experiments/robot/libero/libero_requirements.txt
```

## 2.Dataset
We utilize two datasets for generating adversarial examples:

(a) BridgeData V2:\
Download the BridgeData V2 dataset:
```bash
# Change directory to your base datasets folder
cd roboticAttack

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset datasets/bridge_orig
```

(b) LIBERO: \
Please download [this](https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/main) preprocessed version of the LIBERO dataset, and place it in the `dataset/` folder.

(c) The structure should look like:

    ├── roboticAttack
    │   └── dataset
    |       └──bridge_orig
    |       └──libero_spatial_no_noops
    |       └──libero_object_no_noops
    |       └──libero_goal_no_noops
    |       └──libero_10_no_noops
## 2. Adversarial Patch Generation
(a) Target Manipulation Attack (TMA)
```bash
bash scripts/run_TMA.sh
```

(b) Untargeted Action Discrepancy (UADA)
```bash
bash scripts/run_UADA.sh
```

(b2) DDP ver For UADA
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 VLAAttacker/UADA_wrapper3_ddp.py \
    --maskidx 0 \
    --lr 1e-3 \
    --iter 2000 \
    --MSE_weights 5 \
    --accumulate 1 \                  
    --bs 8 \                            
    --warmup 20 \                        
    --tags XXX \                         
    --geometry True \                    
    --patch_size 3,50,50 \               
    --wandb_project your_wandb_project \ 
    --wandb_entity your_wandb_entity \   
    --innerLoop 50 \                     
    --dataset bridge_orig \              
    --resize_patch False \               
    --reverse_direction True           
```

(c) Untargeted Position-aware Attack (UPA)
```bash
bash scripts/run_UPA.sh
```

---
## 2.Evaluating OpenVLA

```bash
bash scripts/run_simulation.sh
```

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `VLAAttcker/` - Including the code for generating adversarial examples (UADA, UPA, TMA).
+ `scripts/` - Scripts for Attack and Simulation.
+ `experiments/` - Code for evaluating OpenVLA policies in robot environments.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!

---

