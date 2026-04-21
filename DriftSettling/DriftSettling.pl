#!/usr/bin/perl

use strict;
use warnings;

# defining subroutine
sub format_time
{
    # passing argument
    my $minutes = $_[0];
    my $left_minutes = $minutes % 60;
    my $hours = int($minutes / 60);
    my $slurm_format = sprintf("%02d:%02d:00", $hours, $left_minutes);
    my $idefix_format = $minutes / 60;
    my %time = (
        slurm => $slurm_format,
        idefix => $idefix_format
        );
    return %time;
}

my $minutes         = 0.5*60;

my %time_results    = format_time($minutes);
my $IDEFIX_DIR      = "/home/dp316/dp316/dc-fang1/myidefix";                            # The directory where calculations are run
my $folder_name     = "DriftSettling";
my $folder_path     = "/home/dp316/dp316/dc-fang1/IdefixRuns/".$folder_name."/";
my $indir           = $folder_path."inputs/";
my $time            = $time_results{slurm};
my $qos             = "dev";
my $nodes           = "1";
my $gres            = "gpu:4";
my $ntasks_per_node = "4";
my $IDEFIX_EXE      = $folder_path."setup/idefix";
my $options         = "-dec 4 1";
my $name            = "DS";

my @mysubnames = ("test_reflective_alpha1e-2");

my @indexes = (0);

# `mkdir -p $indir`;
for my $index (@indexes) {
print $index."\n";
my $stringdx_1 = $name."_".$mysubnames[$index]; #OW_test
my $outputs_path_1 = $folder_path."outputs/".$stringdx_1; #"/home/dp316/dp316/dc-fang1/IdefixRuns/outputs/OW_test
my $vtksdir1 = $outputs_path_1."/vtks"; #IdefixRuns/outputs/OW_test/vtks
`mkdir -p $vtksdir1`;

##################### .ini file #####################
my $idefix_limit = $time_results{idefix} * 0.99;
`mkdir -p $indir`;
my $inifile = $indir.$stringdx_1.".ini";
open INI, ">$inifile";
print INI <<ENDOFINI;
##
[Grid]
X1-grid    1  1.0  1000  l   20.0
X2-grid    1  1.0471975511965976 512 u 2.0943951023931953 # [pi/3, 2pi/3]
# X2-grid    1  0.0  1024  u  3.141592653589793

[TimeIntegrator]
CFL            0.9
tstop          10000000.0
# tstop            1000.0
first_dt       1.e-6
nstages        2
max_runtime    $idefix_limit

[Hydro]
solver       hllc
csiso         userdef
viscosity   rkl userdef
alpha         1.0e-2

# [Dust]
# nSpecies         3
# drag              tau  1 0.2 0.04
# drag_feedback    false
# DustToGas        0.01
# gamma_i = 1/(rho beta_i) where beta_i is the given value

[Gravity]
potential    central
Mcentral     1.0

[Boundary]
X1-beg    outflow
X1-end    outflow
X2-beg    reflective
X2-end    reflective

[Setup]
sigma0        0.125
sigmaSlope    0.5
csSlope       1.0
h0            0.05

[Output]
vtk        1
dmp_dir    $outputs_path_1
dmp        40
log        1000
vtk_dir    $vtksdir1
# dat_path   $outputs_path_1/timevol.dat
# analysis   1

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
my $script = $indir.$stringdx_1.".sh";
open SCRIPT, ">$script";
print SCRIPT <<ENDOFSCRIPT;
#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=$stringdx_1
#SBATCH --time=$time
#SBATCH --partition=gpu
#SBATCH --qos=$qos

# Request right number of full nodes (48 cores by node for A100-80 GPU nodes))
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH --gres=$gres
#SBATCH --cpus-per-task=1

#SBATCH --account=dp316

cd ${folder_path}setup
# Load the correct modules for the run

module load gcc/9.3.0
module load openmpi/4.1.5-cuda12.3

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

srun --nodes=$nodes --ntasks-per-node=$ntasks_per_node \\
     --hint=nomultithread  --distribution=block:block \\
     /home/dp316/dp316/dc-fang1/scripts/wrapper.sh \\
     $IDEFIX_EXE $options -i $inifile
ENDOFSCRIPT
`chmod u+x $script`;
close SCRIPT;
}
