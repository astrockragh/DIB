#!/bin/bash
#SBATCH --job-name=dib15672_C2v_tau0.05_plateerrs1_balanceerrs0_extratruncation10_emphasizedTcenterTrue
#SBATCH --nodes=1
#SBATCH --account=spergel
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-type=all
#SBATCH --mail-user=cj1223@princeton.edu
#SBATCH --time=2-23:59:59

module purge
module load anaconda3/2022.5
source activate torch-env

# Parameters:
#   --fudge                  1        flux noise inflation factor (1=none)
#   --use_direct             1        1=direct dT; 0=PCA reconstruction
#   --old_errs               0        use original jackknife errors
#   --plate_errs             1        use plate-jackknife errors
#   --flat_prior             0        0=physics priors; 1=flat/uniform priors
#   --fit_spec               1        include DIB absorption profile in fit
#   --fit_dT                 1        include dT profile in fit
#   --cov                    0        0=diagonal covariance; 1=full pixel-pixel
#   --nonlinear_fit          0        nonlinear B/C dipole ratio optimiser (Cs only)
#   --use_scalar_prior       0        prior on B/C scalar ratio for dT
#   --tau_prior              0.05     exp prior slope on lifetime broadening [/cm]
#   --alpha_prior            0.15     Gaussian prior width on vib-rot coupling
#   --extra_truncation       10       extra pixels cropped from each data edge
#   --balance_errs           0        equalise relative DIB/dT uncertainties
#   --err_factor             1.0      noise divisor (>1 tightens likelihood)
#   --fit_peakcenter_offset  0.1      Gaussian prior width on peak centre [/cm]
#   --title                  sweep    label appended to output directory name
#   --nsteps                 20000    MCMC steps per walker
#   --dib                    15672    target DIB rest wavelength [Å]
#   --symmetry_group         C2v      molecular symmetry (Cs or C2v)
#   --emphasize_dT_center    10 3 -2  WIDTH FACTOR OFFSET of dT centre emphasis window

SCRIPT=$HOME/DIB/run_emcee_asym_both_newerrs.py

python $SCRIPT \
    --fudge                  1 \
    --use_direct             1 \
    --old_errs               0 \
    --plate_errs             1 \
    --flat_prior             0 \
    --fit_spec               1 \
    --fit_dT                 1 \
    --cov                    0 \
    --nonlinear_fit          0 \
    --use_scalar_prior       0 \
    --tau_prior              0.05 \
    --alpha_prior            0.15 \
    --extra_truncation       10 \
    --balance_errs           0 \
    --err_factor             1.0 \
    --fit_peakcenter_offset  0.1 \
    --title                  sweep \
    --nsteps                 20000 \
    --dib                    15672 \
    --symmetry_group         C2v \
    --emphasize_dT_center    10 3 -2
