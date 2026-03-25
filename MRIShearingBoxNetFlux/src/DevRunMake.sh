#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=MRIShearingBox1
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --qos=dev

# Request right number of full nodes (48 cores by node for A100-80 GPU nodes))
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1

#SBATCH --account=dp316

# Load the correct modules for the make

module load /mnt/lustre/tursafs1/home/y07/shared/tursa-modules/setup-env
module load gcc/9.3.0
module load cuda/11.4.1
module load openmpi/4.1.1-cuda11.4.1

# Load the CMake modules

module load /home/y07/shared/tursa-modules/setup-env
module load cmake

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

# Build and compile the code (only needed in the first instance)

export IDEFIX_DIR=/home/dp316/dp316/dc-fang1/idefix
cmake $IDEFIX_DIR -DIdefix_MPI=ON -DKokkos_ENABLE_CUDA=ON -D CMAKE_CXX_COMPILER=/mnt/lustre/tursafs1/apps/gcc/9.3.0/bin/gcc -D CMAKE_C_COMPILER=/mnt/lustre/tursafs1/apps/gcc/9.3.0/bin/gcc

make -j 8

# Load the correct modules for the run

module load gcc/9.3.0
module load openmpi/4.1.5-cuda12.3

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

# These will need to be changed to match the actual application you are running
application="./idefix"
# Need to change the dump file requirement later
options="-dec 1 1 4"
# options="-restart -dec 1 1 4"

srun --nodes=1 --ntasks-per-node=4\
     --hint=nomultithread  --distribution=block:block \
     wrapper.sh \
     ${application} ${options}
