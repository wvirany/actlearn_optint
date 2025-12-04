#!/bin/bash

#SBATCH --job-name=exp2
#SBATCH --array=0-14
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/exp2_%A_%a.out

mkdir -p logs

module load python/3.11 scipy-stack r/4.3

export R_LIBS=~/.local/R/$EBVERSIONR/

source ../optint_env/bin/activate

python run_experiments.py --experiment 3 --p 10 --K 1 --shd $1 --trial $SLURM_ARRAY_TASK_ID
