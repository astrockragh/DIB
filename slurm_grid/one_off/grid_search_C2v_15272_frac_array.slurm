#!/bin/bash
#SBATCH --job-name=grid_C2v_15272_frac
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --account=spergel
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2022.5
source activate torch-env

FRACS=(0.95 0.98 0.99 1.0 1.01 1.02 1.05)
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID: frac=$FRAC  cpus=$SLURM_CPUS_PER_TASK"

python -u $HOME/DIB/grid_search_C2v_15272.py \
    --frac   $FRAC \
    --ncpus  $SLURM_CPUS_PER_TASK
