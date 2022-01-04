#!/bin/bash
#
#SBATCH --mail-user=knagaitsev@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/knagaitsev/slurm/out/%j.%N.stdout
#SBATCH --error=/home/knagaitsev/slurm/out/%j.%N.stderr
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=breakout_rl_train
#SBATCH --mem-per-cpu=40000
#SBATCH --time=4:00:00

. /opt/conda/etc/profile.d/conda.sh
conda activate /local/knagaitsev/nn
python train.py
