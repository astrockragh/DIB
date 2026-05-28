#!/bin/bash
#SBATCH --job-name=dib15272_C2v_deltaP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=melchior
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
#SBATCH --time=23:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2022.5
source activate torch-env

SCRATCH_DIB=/scratch/gpfs/MELCHIOR/cj1223/DIB

python -u $HOME/DIB/run_emcee_asym_both_newerrs_sym_alpha_prior_restart.py \
    --dib                    15272 \
    --symmetry_group         C2v \
    --B_not_equal_C          1 \
    --fudge                  1.0 \
    --use_direct             1 \
    --flat_prior             0 \
    --fit_spec               1 \
    --fit_dT                 1 \
    --cov                    0 \
    --nonlinear_fit          0 \
    --tau_prior              0.05 \
    --alpha_prior            0.025 \
    --alpha_prior_cutoff     0 \
    --delta_prior            1 \
    --extra_truncation       10 \
    --balance_errs           0 \
    --emphasize_dT_center    10 2 2 \
    --fit_peakcenter_offset  0.01 \
    --title                  restart_sym_alpha_deltaP \
    --nsteps                 15000 \
    --restart_file           $SCRATCH_DIB/bc_run_SymmetryC2v_BCTrue_F1p0_DTrue_FlatFalse_SpecTrue_dTTrue_covFalse_nonlinFalse_tauSlope0p05_update_prep_restart.h5 \
    --tether_dims            0 1 2 3 7 8 \
    --kde_bandwidth          0.25 \
    --kde_thin               5000
