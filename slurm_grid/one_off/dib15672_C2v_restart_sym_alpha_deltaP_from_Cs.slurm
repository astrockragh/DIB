#!/bin/bash
#SBATCH --job-name=dib15672_C2v_dP_fromCs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=melchior
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1-23:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2022.5
source activate torch-env

SCRATCH_DIB=/scratch/gpfs/MELCHIOR/cj1223/DIB

python -u $HOME/DIB/run_emcee_asym_both_newerrs_sym_alpha_prior_restart.py \
    --dib                    15672 \
    --symmetry_group         C2v \
    --B_not_equal_C          1 \
    --fudge                  1.0 \
    --use_direct             1 \
    --flat_prior             0 \
    --fit_spec               1 \
    --fit_dT                 1 \
    --cov                    0 \
    --nonlinear_fit          0 \
    --tau_prior              0.15 \
    --alpha_prior            0.075 \
    --alpha_prior_cutoff     0 \
    --delta_prior            1 \
    --extra_truncation       10 \
    --balance_errs           0 \
    --emphasize_dT_center    10 2 -2 \
    --fit_peakcenter_offset  0.15 \
    --title                  restart_sym_alpha_deltaP_from_Cs \
    --nsteps                 20000 \
    --restart_file           "$SCRATCH_DIB/15672_run_DIB15672_SymCs_BCTrue_F1p0_DTrue_FlatFalse_SpecTrue_dTTrue_covFalse_nonlinFalse_tauSlope0p15_alphaSlope0p075_alphaCutFalse_errDefault_trunc10_balanceErrFalse_dTcent[10p0,2p0,-2p0]_wavoffset0p1_dPTrue_restart_sym_alpha_deltaP.h5" \
    --tether_dims            0 1 2 3 7 8 \
    --kde_bandwidth          0.25 \
    --kde_thin               5000
