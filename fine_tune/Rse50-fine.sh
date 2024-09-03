#!/bin/bash
#SBATCH --job-name=Res50-fine
#SBATCH --account=Project_2010727
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "Res50-fine.txt"
#SBATCH -e "Res50-fine.txt"


# squeue -l -u $USER
# scancel $(squeue -u $USER -n Res50-fine -o "%.18i" | tail -n 1)
# sbatch Rse50-fine.sh
# chmod +x Res50-fine.sh
# cd '/projappl/project_2010727/Transfer Learning/fine_tune'
# /projappl/project_2010727/taxonomist/Tykky/bin/tensorboard --logdir tb_logs --port 6009

export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"

EXPERIMENT_NAME="Fin1.3, Res50, S=10%, TE=20, TLR=0.01, FTLR=0.001"


python Res50-fine.py \
--experiment_name "$EXPERIMENT_NAME" \
--transfer_learning_rate 0.01 \
--transfer_epochs 20 \
--fine_tuning_learning_rate 0.001 \
--fine_tuning_epochs 50
