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
#SBATCH --partition=am-148-s20
#SBATCH --account=am-148-s20
#SBATCH --qos=am-148-s20
module load cuda10.1

srun match
