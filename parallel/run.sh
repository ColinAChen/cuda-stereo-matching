#!/bin/bash
#note-there can be no line spaces between # SBATCH directives .
#SBATCH --job-name=parallel_match
#SBATCH --output=out %j. out
#SBATCH --error=out_error % j. err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=coachen@ucsc.edu
#SBATCH --partition=gpuq
#SBATCH --account=gpuq
#SBATCH --qos=gpuq
module load cuda10.1

srun match
