#!/bin/bash
#SBATCH --job-name=test_15672
#SBATCH --nodes=1
#SBATCH --account=spergel
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --time=71:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2022.5
source activate torch-env

python ../run_emcee_asym_both.py \
    --dib            15672         \
    --symmetry_group Cs           \
    --B_not_equal_C  1             \
    --nsteps         20000         \
    --fudge          1             \
    --use_direct     1             \
    --plate_errs     1             \
    --flat_prior     0             \
    --fit_spec       1             \
    --fit_dT         1             \
    --cov            0             \
    --nonlinear_fit  0             \
    --use_scalar_prior 0           \
    --tau_prior      0.15          \
    --alpha_prior    0.15          \
    --extra_truncation 10           \
    --balance_errs   1             \
    --fit_peakcenter_offset 0.15    \
    --title          test
