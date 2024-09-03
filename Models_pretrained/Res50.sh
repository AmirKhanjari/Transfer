#!/bin/bash
#SBATCH --job-name=Res50
#SBATCH --account=Project_2010727
#SBATCH --partition=gpu
#SBATCH --time=02:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "Res50.txt"
#SBATCH -e "Res50.txt"



# squeue -l -u $USER
# scancel $(squeue -u $USER -n Res50 -o "%.18i" | tail -n 1)
# sbatch Res50.sh
# chmod +x Res50.sh
# cd /projappl/project_2010727/Models_pretrained
# /projappl/project_2010727/taxonomist/Tykky/bin/tensorboard --logdir tb_logs --port 6006


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"

EXPERIMENT_NAME="Fin1.3, Res50, 100%, fold=0, epochs4_RAW_lr0.001, FP, lr=0.001"

python Res50.py \
 --experiment_name "$EXPERIMENT_NAME" \
 --learning_rate 0.001




