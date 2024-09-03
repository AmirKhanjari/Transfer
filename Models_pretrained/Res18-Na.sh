#!/bin/bash
#SBATCH --job-name=Res18-Na
#SBATCH --account=Project_2010727
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "Res18-Na.txt"
#SBATCH -e "Res18-Na.txt"



# squeue -l -u $USER
# scancel $(squeue -u $USER -n Res18-Na -o "%.18i" | tail -n 1)
# sbatch Res18-Na.sh
# chmod +x Res18-Na.sh
# cd '/projappl/project_2010727/Transfer Learning/Models_pretrained'
# /projappl/project_2010727/taxonomist/Tykky/bin/tensorboard --logdir tb_logs --port 6006


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"

EXPERIMENT_NAME="Fin1.1, Res, 100%, fold=0, GS, lr=0.0001"

python Res18-Na.py \
 --experiment_name "$EXPERIMENT_NAME" \
 --learning_rate 0.0001




