#!/usr/bin/perl

use strict;
use warnings;

# The directory where calculations are run
my $IDEFIX_DIR="/home/dp316/dp316/dc-fang1/myidefix";
my $indir = $IDEFIX_DIR."/david/MRIShearingBoxNetFlux/";
my $outdir = "/home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/";
my $name = "MRI";
my $time   = "00:30:00";
my $nodes = "1";
my $gres = "gpu:4";
my $ntasks_per_node= "4";
my $application = "./idefix";
my $options = "-dec 1 1 4";
my @myindex = ("mode_n2_res2x", "mode_n4_res2x", "mode_n6_res2x", "mode_n8_res2x", "mode_n10_res2x", "mode_n12_res2x");
my @my_n = (2, 4, 6, 8, 10, 12);
my @vx_r = (0, 0, 0, 0, 0, 0.19);
my @vx_i = (0.41, 0.47, 0.44, -0.35, -0.21, 0);
my @vy_r = (0,0,0,0,0,0);
my @vy_i = (0.14, 0.3, 0.42, -0.53, -0.61,-0.63 );
my @Bx_r = (-0.29, -0.44, -0.55, 0.64, 0.72, 0.72);
my @Bx_i = (0,0,0,0,0,0);
my @By_r = (0.85,0.7,0.57,-0.43,-0.25,0);
my @By_i = (0,0,0,0,0,-0.22);

# Loop starts here
#my $maxindex = @myindex;
#my $maxindex = 3;
my @indexes = (0);
for my $index (@indexes) {
print $index;
my $stringdx = $name."_".$myindex[$index];
my $output1 = $outdir.$stringdx;
my $vtksdir1 = $output1."/vtks";
`mkdir -p $output1`;
`mkdir -p $vtksdir1`;
my $input1 = $indir."in/";
`mkdir -p $input1`;

##################### .ini file #####################
my $inifile = $input1.$stringdx.".ini";
open INI, ">$inifile";
print INI <<ENDOFINI;
##
[Grid]

X1-grid    1    -0.5   128   u    0.5
X2-grid    1    -1.57080   100  u    1.57080
X3-grid    1    -0.5   128   u    0.5


[TimeIntegrator]

CFL              0.8
CFL_max_var      1.1
tstop            130
first_dt         1.e-4
nstages          2

[Hydro]

csiso          constant 1.0
solver         hlld
rotation       1.0
shearingBox    -1.5

[Gravity]
bodyForce      userdef

[Boundary]

X1-beg        shearingbox
X1-end        shearingbox
X2-beg        periodic
X2-end        periodic
X3-beg        periodic
X3-end        periodic

[Setup]

v0           0.00001
# Choice of amplitude
B0           0.05
n	         $my_n[$index]
vx-r         $vx_r[$index]
vx-i         $vx_i[$index]
vy-r         $vy_r[$index]
vy-i         $vy_i[$index]
Bx-r         $Bx_r[$index]
Bx-i         $Bx_i[$index]
By-r         $By_r[$index]
By-i         $By_i[$index]


[Output]

dmp        100
vtk        3.0
vtk_slice1 1.0  1  0.0  cut
analysis   0.1
vtk_dir    $vtksdir1
dat_dir	 $output1/timevol.dat
# File produced automatically by a Perl script
# Do not edit
ENDOFINI

# Batch script
my $script = $input1.$stringdx.".sh";
open SCRIPT, ">$script";
print SCRIPT <<ENDOFSCRIPT;
#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=$stringdx
#SBATCH --time=$time
#SBATCH --partition=gpu
#SBATCH --qos=dev

# Request right number of full nodes (48 cores by node for A100-80 GPU nodes))
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH --gres=$gres
#SBATCH --cpus-per-task=1

#SBATCH --account=dp316

cd $indir
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

# export IDEFIX_DIR=$IDEFIX_DIR
# cmake $IDEFIX_DIR -DIdefix_MPI=ON -DKokkos_ENABLE_CUDA=ON -D CMAKE_CXX_COMPILER=/mnt/lustre/tursafs1/apps/gcc/9.3.0/bin/gcc -D CMAKE_C_COMPILER=/mnt/lustre/tursafs1/apps/gcc/9.3.0/bin/gcc

# make -j 8

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

srun --nodes=$nodes --ntasks-per-node=$ntasks_per_node \\
     --hint=nomultithread  --distribution=block:block \\
     wrapper.sh \\
     $application $options -i $inifile
ENDOFSCRIPT
`chmod u+x $script`;
close SCRIPT;
}
