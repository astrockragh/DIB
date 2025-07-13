#!/bin/bash
#SBATCH --job-name=flat  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --account=astro           # account
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=160:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2022.5
source activate torch-env
python run_emcee_asym_more_options.py --A_not_equal_B True --fudge 5 --use_direct True --flat_prior True --fit_spec True --fit_dT True --cov False