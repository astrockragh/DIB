#!/bin/bash
#SBATCH --job-name=dib15272_Cs_alpha0.03_nowall
#SBATCH --nodes=1
#SBATCH --account=spergel
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --time=5-23:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2022.5
source activate torch-env

python -u $HOME/DIB/run_emcee_asym_both_newerrs_sym_alpha_prior.py \
    --dib                    15272 \
    --symmetry_group         Cs \
    --B_not_equal_C          1 \
    --fudge                  1.0 \
    --use_direct             1 \
    --flat_prior             0 \
    --fit_spec               1 \
    --fit_dT                 1 \
    --cov                    0 \
    --nonlinear_fit          0 \
    --tau_prior              0.15 \
    --alpha_prior            0.03 \
    --alpha_prior_cutoff     0 \
    --extra_truncation       10 \
    --balance_errs           0 \
    --emphasize_dT_center    10 3 2 \
    --fit_peakcenter_offset  0.01 \
    --title                  nowall \
    --nsteps                 25000
