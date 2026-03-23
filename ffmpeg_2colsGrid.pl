#!/usr/bin/perl

use strict;
use warnings;

my $root = "/home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/";
my @folders = (
    "MRI_mode_n2", "MRI_mode_n2_res2x", 
    "MRI_mode_n4", "MRI_mode_n4_res2x", 
    "MRI_mode_n6", "MRI_mode_n6_res2x", 
    "MRI_mode_n8", "MRI_mode_n8_res2x", 
    "MRI_mode_n10", "MRI_mode_n10_res2x", 
    "MRI_mode_n12", "MRI_mode_n12_res2x"
);

my $cmdfile = $root . "ffmpeg_2colsGrid.sh";
open(my $fh, '>', $cmdfile) or die "Impossible de créer $cmdfile: $!";

# 1. Préparation des entrées (-i ...)
my @inputs = map { "-i $root$_/videos/slice1.mp4" } @folders;

# 2. Génération dynamique du LAYOUT pour xstack (2 colonnes)
my @layout_parts;
for my $i (0 .. $#folders) {
    my $column = $i % 2;       # 0 pour gauche, 1 pour droite
    my $row    = int($i / 2);  # Lignes 0, 1, 2, 3, 4, 5
    
    my $x = ($column == 0) ? "0" : "w0";
    
    my $y;
    if ($row == 0) {
        $y = "0";
    } else {
        # On additionne les hauteurs des vidéos de la première colonne (indices pairs)
        # pour définir le décalage vertical : h0, h0+h2, h0+h2+h4...
        my @heights = map { "h" . ($_ * 2) } (0 .. $row - 1);
        $y = join('+', @heights);
    }
    push @layout_parts, "${x}_${y}";
}

# IMPORTANT : On utilise les indices directs [0][1][2]... au lieu de [v0][v1]...
my $inputs_labels = join('', map { "[$_]" } (0 .. $#folders));
my $layout_string = join('|', @layout_parts);

# 3. Écriture finale dans le fichier .sh
print $fh "#!/bin/bash\n\n";
print $fh "ffmpeg \\\n";
print $fh join(" \\\n", @inputs) . " \\\n";
print $fh "-filter_complex \"\\\n";
print $fh "$inputs_labels xstack=inputs=" . scalar(@folders) . ":layout=$layout_string" . "[v]\" \\\n";
print $fh "-map \"[v]\" -c:v libx264 -crf 23 -preset veryfast -y $root" . "output_grid.mp4\n";

close $fh;

# Rendre le script exécutable
system("chmod u+x $cmdfile");
print "[OK] Generated : $cmdfile\n";