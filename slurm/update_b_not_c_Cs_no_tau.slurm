#!/bin/bash
#SBATCH --job-name=no_tau  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --account=astro           # account
#SBATCH --cpus-per-task=64        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=48:59:59         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2022.5
source activate torch-env
python ../run_emcee_asym_more_options_fast.py --B_not_equal_C 1 -fudge 1 -use_direct 1 -tau_prior 0.0001 --alpha_prior 0.1 -flat_prior 0 -fit_spec 1 -fit_dT 1 -cov 0 -symmetry_group Cs -nsteps 10000 -use_scalar_prior 0 --nonlinear_fit 0 -title update_notau