#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=OW_test2
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --qos=dev

# Request right number of full nodes (48 cores by node for A100-80 GPU nodes))
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1

#SBATCH --account=dp316

cd /home/dp316/dp316/dc-fang1/outputs/OhmicWind/
# Load the correct modules for the run

module load gcc/9.3.0
module load openmpi/4.1.5-cuda12.3

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

# These will need to be changed to match the actual application you are running
# application="./idefix"
# Need to change the dump file requirement later
# options="-dec 1 1 4"
# options="-restart -dec 1 1 4"

srun --nodes=1 --ntasks-per-node=4 \
     --hint=nomultithread  --distribution=block:block \
     /home/dp316/dp316/dc-fang1/scripts/wrapper.sh \
     /home/dp316/dp316/dc-fang1/myidefix/david/OhmicWind/idefix -dec 4 1 -i /home/dp316/dp316/dc-fang1/outputs/OhmicWind/in/OW_test2.ini
