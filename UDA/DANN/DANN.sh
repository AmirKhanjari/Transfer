#!/bin/bash
#SBATCH --job-name=DANN
#SBATCH --account=Project_2010727
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=373G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "DANN.txt"
#SBATCH -e "DANN.txt"

export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"

python /projappl/project_2010727/Jindongwang/code/DeepDA/main.py \
  --backbone resnet18 \
  --transfer_loss_weight 1.0 \
  --transfer_loss adv \
  --lr 0.01 \
  --weight_decay 0.001 \
  --batch_size 64 \
  --momentum 0.9 \
  --lr_scheduler True \
  --lr_gamma 0.001 \
  --lr_decay 0.75 \
  --n_iter_per_epoch 200 \
  --n_epoch 90 \
  --seed 1 \
  --num_workers 40

# squeue -l -u $USER
# scancel $(squeue -u $USER -n DANN -o "%.18i" | tail -n 1)
# sbatch /projappl/project_2010727/Jindongwang/code/DeepDA/DANN/DANN.sh
