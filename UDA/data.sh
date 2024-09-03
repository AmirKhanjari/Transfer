#!/bin/bash
#SBATCH --job-name=data
#SBATCH --account=Project_2010727
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=373G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "data.txt"
#SBATCH -e "data.txt"


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"




python data_loader.py

# squeue -l -u $USER
# scancel $(squeue -u $USER -n data -o "%.18i" | tail -n 1)
# sbatch data.sh
# cd /projappl/project_2010727/Jindongwang/code/DeepDA/
