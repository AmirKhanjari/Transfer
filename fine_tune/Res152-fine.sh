#!/bin/bash
#SBATCH --job-name=Res152-fine
#SBATCH --account=Project_2010727
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "Res152-fine.txt"
#SBATCH -e "Res152-fine.txt"


# squeue -l -u $USER
# scancel $(squeue -u $USER -n Res152-fine -o "%.18i" | tail -n 1)
# sbatch Res152-fine.sh
# chmod +x Res152-fine.sh
# cd /projappl/project_2010727/fine_tune
# /projappl/project_2010727/taxonomist/Tykky/bin/tensorboard --logdir tb_logs --port 6009

export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"

EXPERIMENT_NAME="Fin1.3, Res152, S=10%, TE=10, TLR=0.001, FTLR=0.00001"


python Res152-fine.py \
--experiment_name "$EXPERIMENT_NAME" \
--transfer_learning_rate 0.001 \
--transfer_epochs 10 \
--fine_tuning_learning_rate 0.00001 \
--fine_tuning_epochs 50
