#!/bin/bash
#SBATCH -A b1140
#SBATCH --partition=buyin
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=6G
#SBATCH --array=1-4
#SBATCH --mail-user=andrewli2026@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=silent_synapses

module purge all
module load python/3.10.1
source /home/alz2328/.cache/pypoetry/virtualenvs/silent-synapses-bCxPY3Gs-py3.10/bin/activate

echo "Task ID $SLURM_ARRAY_TASK_ID"
python3 main.py
