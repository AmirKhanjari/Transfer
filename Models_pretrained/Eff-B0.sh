#!/bin/bash
#SBATCH --job-name=Eff-B0
#SBATCH --account=Project_2010727
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "Eff-B0.txt"
#SBATCH -e "Eff-B0.txt"



# squeue -l -u $USER
# scancel $(squeue -u $USER -n Eff-B0 -o "%.18i" | tail -n 1)
# sbatch Eff-B0.sh
# chmod +x Eff-B08.sh
# cd '/projappl/project_2010727/Transfer Learning/Models_pretrained'
# /projappl/project_2010727/taxonomist/Tykky/bin/tensorboard --logdir tb_logs --port 6006


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"

EXPERIMENT_NAME="Data= Fin1.1, SS= 100%, Model= Eff, Imagenet, Finpre"

python Eff-B0.py \
 --experiment_name "$EXPERIMENT_NAME" \
 --learning_rate 0.001
