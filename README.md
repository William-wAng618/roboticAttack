# Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics


[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://www.arxiv.org/abs/2411.13587)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

<div align="center">
  <img src=".\fig\mainfig.png">
</div>
<p>
Overall Adversarial Framework. The robot captures an input image, processes it through a vision-language model to generate
tokens representing actions, and then uses an action de-tokenizer for discrete bin prediction. The model is optimized with adversarial
objectives focusing on various discrepancies and geometries (i.e., UADA, UPA, TMA). Forward propagation is shown in black, and
backpropagation is highlighted in pink. These objectives aim to maximize errors and minimize task performance, with visual emphasis on
3D-space manipulation and a focus on generating adversarial perturbation δ during task execution, such as picking up a can.
</p>

Built on top of [OpenVLA](https://github.com/openvla/openvla), a remarkable generalist vision-language-action model work. 

---

## Latest Updates
- [2024-11-26] Pre release


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



