#!/bin/bash
#SBATCH --job-name=Res152
#SBATCH --account=Project_2010727
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "Res152.txt"
#SBATCH -e "Res152.txt"



# squeue -l -u $USER
# scancel $(squeue -u $USER -n Res152 -o "%.18i" | tail -n 1)
# sbatch Res152.sh
# chmod +x Res152.sh
# cd /projappl/project_2010727/Models_pretrained
# /projappl/project_2010727/taxonomist/Tykky/bin/tensorboard --logdir tb_logs --port 6006


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"

EXPERIMENT_NAME="Fin1.3, Res152, 100%, fold=0, epochs4_lr0.001, FP, lr=0.0001"

python Res152.py \
 --experiment_name "$EXPERIMENT_NAME" \
 --learning_rate 0.0001




