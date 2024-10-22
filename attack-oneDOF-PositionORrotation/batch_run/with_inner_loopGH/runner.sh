#!/bin/bash
#SBATCH -J openvla_attack_test
#SBATCH -t 48:00:00
#SBATCH --output=/home/tw9146/tw/openvla-main/run/white_patch_attack/slurm_out/%x_%j.out
#SBATCH --error=/home/tw9146/tw/openvla-main/run/white_patch_attack/slurm_err/%x_%j.err
#SBATCH -p grace
#SBATCH -A domainacs
#SBATCH --mail-user=tw9146@rit.edu
#SBATCH --mem=100G
#SBATCH --gres=gpu:gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#export CUDA_HOME=/.autofs/tools/spack/opt/spack/linux-rhel9-neoverse_v2/gcc-12.3.1/cuda-12.4.0-5ncc6rpt2kvuwx5iomvpkdadb5oxvb6s
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#spack load cuda@12.4
export PATH=$HOME/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.3/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$HOME/cuda-12.3
#spack load /ytea2go # 11.8
#spack load /5ncc6rp # 12.4:
#spack load /5ncc6rp # 12.4:
#spack load /46r35gb
#spack load /hz35fea # aarch gcc
#spack load gcc
source ~/aarchconda/aarchconda/etc/profile.d/conda.sh
conda activate openvla_gh
export WANDB_API_KEY='2afe2846a028d2bc651361a4fbdf85cfbd9d9eab'
wandb login

bash attack-oneDOF-PositionORrotation/batch_run/with_inner_loopGH/attack_white_patch0.sh
#bash attack-oneDOF-PositionORrotation/batch_run/with_inner_loopGH/attack_white_patch0.sh &
#bash attack-oneDOF-PositionORrotation/batch_run/with_inner_loopGH/attack_white_patch1.sh &
