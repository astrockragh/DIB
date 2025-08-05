#!/bin/bash
#SBATCH --job-name=pgo_asym   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --account=astro                # account
#SBATCH --cpus-per-task=64        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=6:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=cj1223@princeton.edu

module purge
bash pgopher_asymtop.sh