#!/bin/bash
#SBATCH --job-name=pred
#SBATCH --account=Project_2010727
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_pred.txt"
#SBATCH -e "e_pred.txt"


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"



python Transfermonitor.py

# squeue -l -u $USER
# scancel $(squeue -u $USER -n pred -o "%.18i" | tail -n 1)
# sbatch Transfermonitor.sh
# chmod +x Transfermonitor.sh