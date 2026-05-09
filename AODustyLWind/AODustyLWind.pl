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

my $minutes         = 240;
my $gpus            = 1;

my %time_results    = format_time($minutes);
my $IDEFIX_DIR      = "/home/dp316/dp316/dc-fang1/IdefixGeoffroy";                            # The directory where calculations are run
my $folder_name     = "AODustyLWind";
my $folder_path     = "/home/dp316/dp316/dc-fang1/IdefixRuns/".$folder_name."/";
my $indir           = $folder_path."inputs/";
my $time            = $time_results{slurm};
my $qos             = "dev";
my $nodes           = "1";
my $gres            = "gpu:$gpus";
my $ntasks_per_node = $gpus;
my $setup_dir      = $folder_path."reload_l";
my $IDEFIX_EXE      = $setup_dir."/idefix";
my $options         = "-dec $gpus 1";
my $name            = "clean";

my @mysubnames = ("reload_wind");

my @indexes = (0);

for my $index (@indexes) {
print $index."\n";
my $stringdx_1 = $name."_".$mysubnames[$index]; #OW_test
my $outputs_path_1 = $folder_path."outputs/".$stringdx_1; #"/home/dp316/dp316/dc-fang1/IdefixRuns/outputs/OW_test
my $vtksdir1 = $outputs_path_1."/vtks"; #IdefixRuns/outputs/OW_test/vtks
`mkdir -p $vtksdir1`;

##################### .ini file #####################
my $idefix_limit = $time_results{idefix} * 0.99;
print $indir."\n";
`mkdir -p $indir`;
my $inifile = $indir.$stringdx_1.".ini";
open INI, ">$inifile";
print INI <<ENDOFINI;
##
[Grid]
X1-grid    1  1.0  512  l   20.0
X2-grid    3  0.0  256   u  1.28   96  u  1.861592653589  256  u  3.141592653589793
# X2-grid    1  0.0  1024  u  3.141592653589793

[TimeIntegrator]
CFL            0.9
tstop          500.0
# tstop            1000.0
first_dt       1.e-6
nstages        2
max_runtime    $idefix_limit

[Hydro]
solver       hlld
ambipolar    explicit  userdef
resistivity  explicit  userdef
gamma        1.0001


# [Dust]
# nSpecies         1
# drag             userdef  1   1e-4    # St=1, 0.2, 0.04
# drag_feedback    no

[Particles]
count            per_proc  20
stopping_time    constant  1e-5
ParticleMass     3e-3
# DustToGas        3e-3


[Gravity]
potential    central
Mcentral     1.0

[Boundary]
X1-beg    userdef
X1-end    userdef
X2-beg    userdef
X2-end    userdef

[Setup]
Rm0                    10.0
etab0                  1.0
epsilon                0.05
beta                   1000
# beta->10000?
epsilonTop             0.3
Hideal                 5.0
Am                     1.0
densityFloor           1.0e-7
transitionSmoothing    0.5
# fromDump               true
reload_path             /home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/reload_l/clean_wind.0001.dmp

[Output]
uservar    eta    Am    InvDt
vtk        0.2
dmp_dir    $outputs_path_1
dmp        200
log        1000
vtk_dir    $vtksdir1
dat_path   $outputs_path_1/timevol.dat
analysis   1

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
