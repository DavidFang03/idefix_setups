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

my $minutes         = 20;
my $gpus            = 1;

my %time_results    = format_time($minutes);
# my $IDEFIX_DIR      = "/home/dp316/dp316/dc-fang1/myidefix";                            # The directory where calculations are run
# my $IDEFIX_DIR      = "/home/dp316/dp316/dc-fang1/myidefix";                            # The directory where calculations are run
my $IDEFIX_DIR      = "/home/dp316/dp316/dc-fang1/IdefixGeoffroy";                            # The directory where calculations are run
my $folder_name     = "VerticalSettling";
my $folder_path     = "/home/dp316/dp316/dc-fang1/IdefixRuns/".$folder_name."/";
my $indir           = $folder_path."inputs/";
my $time            = $time_results{slurm};
my $qos             = "dev";
my $nodes           = "1";
my $gres            = "gpu:".$gpus;
my $ntasks_per_node = $gpus;
my $setup_dir      = $folder_path."setup_l";
my $IDEFIX_EXE      = $setup_dir."/idefix";
my $options         = "-dec 1 1 1" ;
my $name            = "SettlingL";
# my $name            = "Settling";

my @mysubnames = ("Tau");

my @indexes = (0);

# `mkdir -p $indir`;
for my $index (@indexes) {
print $index."\n";
my $stringdx_1 = $name."_".$mysubnames[$index]; #OW_test
my $outputs_path_1 = $folder_path."outputs/".$stringdx_1; #"/home/dp316/dp316/dc-fang1/IdefixRuns/outputs/OW_test
my $vtksdir1 = $outputs_path_1."/vtks"; #IdefixRuns/outputs/OW_test/vtks
`mkdir -p $vtksdir1`;
print $vtksdir1."\n";

##################### .ini file #####################
my $idefix_limit = $time_results{idefix} * 0.99;
`mkdir -p $indir`;
my $inifile = $indir.$stringdx_1.".ini";
open INI, ">$inifile";
print INI <<ENDOFINI;
##
[Grid]
X1-grid    1  1.99      1  u  2.01
X2-grid    1  -0.0125      1  u  0.0125
X3-grid    1  -0.15  512    u  0.3

[TimeIntegrator]
CFL         0.5
tstop       20
first_dt    1.0e-5
nstages     2

[Hydro]
solver    hllc
csiso     userdef
# viscosity    explicit  userdef

# [Dust]
# nSpecies         3
# drag             tau  1   0.2   0.04    # St=1, 0.2, 0.04
# drag_feedback    no

[Particles]
count            per_proc  1
stopping_time    constant  1.0
ParticleMass     3e-3
DustToGas        3e-3

[Gravity]
potential    central 
Mcentral     1.0

[Boundary]
X1-beg    userdef
X1-end    userdef
X2-beg    periodic
X2-end    periodic
X3-beg    outflow
X3-end    outflow

[Setup]
sigma0        0.125
sigmaSlope    -0.5
CsSlope       -0.5
h0            0.05
alpha         1.0e-4

[Output]
vtk    0.1
dmp    1000.0
log        1000
dmp_dir    $outputs_path_1
vtk_dir    $vtksdir1
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

cd $setup_dir
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
