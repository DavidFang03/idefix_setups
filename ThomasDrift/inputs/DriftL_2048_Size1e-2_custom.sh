#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=DriftL_2048_Size1e-2_custom
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --qos=dev

# Request right number of full nodes (48 cores by node for A100-80 GPU nodes))
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

#SBATCH --account=dp316

cd /home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/setup_l
# Load the correct modules for the run

module load gcc/9.3.0
module load openmpi/4.1.5-cuda12.3

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

srun --nodes=1 --ntasks-per-node=1 \
     --hint=nomultithread  --distribution=block:block \
     /home/dp316/dp316/dc-fang1/scripts/wrapper.sh \
     /home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/setup_l/idefix -dec 1 -i /home/dp316/dp316/dc-fang1/IdefixRuns/ThomasDrift/inputs/DriftL_2048_Size1e-2_custom.ini
