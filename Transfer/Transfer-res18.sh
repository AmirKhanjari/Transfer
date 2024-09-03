#!/bin/bash
#SBATCH --job-name=Transfer-res18
#SBATCH --account=Project_2010727
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "TRes.txt"
#SBATCH -e "TRes.txt"



# squeue -l -u $USER
# scancel $(squeue -u $USER -n Transfer-res18 -o "%.18i" | tail -n 1)
# sbatch Transfer-res18.sh
# chmod +x Transfer-res18.sh
# cd /projappl/project_2010727/Transfer
# /projappl/project_2010727/taxonomist/Tykky/bin/tensorboard --logdir tb_logs --port 6006


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"

EXPERIMENT_NAME="Fin1.1, IT, lr=0.01"

python Transfer-res18.py \
 --experiment_name "$EXPERIMENT_NAME" \
 --learning_rate 0.01




 