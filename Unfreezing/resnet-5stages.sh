#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --account=Project_2010727
#SBATCH --partition=gpu
#SBATCH --time=01:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "augresnet-5stages.txt"
#SBATCH -e "augrsnet-5stages.txt"


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"



python resnet-5stages.py

# squeue -l -u $USER
# scancel $(squeue -u $USER -n resnet -o "%.18i" | tail -n 1)
# sbatch resnet-5stages.sh
# chmod +x resnet-5stages.sh