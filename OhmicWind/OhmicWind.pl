#!/usr/bin/perl

use strict;
use warnings;

# The directory where calculations are run
my $IDEFIX_DIR="/home/dp316/dp316/dc-fang1/myidefix";
# my $SETUP_DIR="/home/dp316/dp316/dc-fang1/myidefix/david/OhmicWind";
my $outdir = "/home/dp316/dp316/dc-fang1/outputs/OhmicWind/";
my $indir = $outdir."in/";
my $name = "OW";
my $time   = "00:30:00";
my $nodes = "1";
my $gres = "gpu:4";
my $ntasks_per_node= "4";
my $IDEFIX_EXE="/home/dp316/dp316/dc-fang1/myidefix/david/OhmicWind/idefix";
my $options = "-dec 1 4";
my @mysubnames = ("test");

my @indexes = (0);
# Loop starts here
#my $maxindex = @myindex;
#my $maxindex = 3;

`mkdir -p $indir`;
for my $index (@indexes) {
print $index;
my $stringdx = $name."_".$mysubnames[$index]; #OW_test
my $output1 = $outdir.$stringdx;
my $vtksdir1 = $output1."/vtks"; #OW_test/vtks
`mkdir -p $vtksdir1`;

##################### .ini file #####################
my $inifile = $indir.$stringdx.".ini";
open INI, ">$inifile";
print INI <<ENDOFINI;
##
[Grid]
X1-grid    1  1.0  768  l   100.0
X2-grid    3  0.0  64   s+  1.28   96  u  1.861592653589  64  s-  3.141592653589793

[TimeIntegrator]
CFL            0.9
tstop          10000000.0
# tstop            1000.0
first_dt       1.e-6
nstages        2
max_runtime    19.8

[Hydro]
solver       hlld
resistivity    explicit  userdef
gamma        1.0001

[Gravity]
potential    central
Mcentral     1.0

[Boundary]
X1-beg    userdef
X1-end    userdef
X2-beg    axis
X2-end    axis

[Setup]
epsilon                0.05
beta                   1000
epsilonTop             0.3
Hideal                 5.0
densityFloor           1.0e-7
transitionSmoothing    0.5

[Output]
uservar    InvDt
vtk        0.1
log        1000
vtk_dir    $vtksdir1
dat_path   $output1/timevol.dat
analysis   0.1

# [Output]
# 
# dmp        100
# vtk        3.0
# vtk_slice1 1.0  1  0.0  cut
# vtk_dir    $vtksdir1
# File produced automatically by a Perl script
# Do not edit
ENDOFINI

# Batch script
my $script = $indir.$stringdx.".sh";
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

cd $outdir
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
     /home/dp316/dp316/dc-fang1/scripts/wrapper.sh \\
     $IDEFIX_EXE $options -i $inifile
ENDOFSCRIPT
`chmod u+x $script`;
close SCRIPT;
}
