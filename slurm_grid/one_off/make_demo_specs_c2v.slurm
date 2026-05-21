#!/bin/bash
#SBATCH --job-name=verbose_demo_specs
#SBATCH --nodes=1
#SBATCH --account=spergel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2022.5
source activate torch-env

python -u $HOME/DIB/make_demo_specs.py
