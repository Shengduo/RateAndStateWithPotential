#!/bin/bash

# Job name
#SBATCH --job-name="Trial_0216_combined"

# Number of processor cores / tasks
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH --nodes=1

# Number of GPUs
#SBATCH --gres=gpu:2

# Wall time : maximum allowed run time
#SBATCH --time=168:00:00  
#SBATCH --qos=normal

# Send email to user
#SBATCH --mail-user=sliu5@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Run the mpi job
python TuneDimXi_logV_WDsep_deltaTSqed_combinedSet.py &>> log/Trial0216_combined_800_2_hpc
