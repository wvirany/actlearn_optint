#!/bin/bash

#SBATCH --job-name=exp1
#SBATCH --array=0-14
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/exp1_%A_%a.out

mkdir -p logs

module load python/3.11 scipy-stack r/4.3

export R_LIBS=~/.local/R/$EBVERSIONR/

source ../optint_env/bin/activate

python run_experiments.py --experiment 1 --p $1 --trial $SLURM_ARRAY_TASK_ID
