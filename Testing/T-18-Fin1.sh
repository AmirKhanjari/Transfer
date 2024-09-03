#!/bin/bash
#SBATCH --job-name=T-18-Fin1
#SBATCH --account=Project_2010727
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=373G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "T-18-Fin1.txt"
#SBATCH -e "T-18-Fin1.txt"


export PATH="/projappl/project_2010727/taxonomist/Tykky/bin:$PATH"



python T-18-Fin1.py

# squeue -l -u $USER
# scancel $(squeue -u $USER -n T-18-Fin1 -o "%.18i" | tail -n 1)
# sbatch T-18-Fin1.sh
# chmod +x T-18-Fin1.sh
# cd '/projappl/project_2010727/Transfer Learning/Testing'