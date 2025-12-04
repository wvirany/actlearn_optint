#!/bin/bash

#SBATCH --job-name=exp1
#SBATCH --array=0-14
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/

mkdir -p logs

source ../optint/bin/activate

python run_experiments.py --experiment 1 --p $1 --trial $SLURM_ARRAY_TASK_ID