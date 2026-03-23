#!/bin/bash

ffmpeg \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n2/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n2_res2x/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n4/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n4_res2x/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n6/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n6_res2x/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n8/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n8_res2x/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n10/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n10_res2x/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n12/videos/slice1.mp4 \
-i /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/MRI_mode_n12_res2x/videos/slice1.mp4 \
-filter_complex "\
[0][1][2][3][4][5][6][7][8][9][10][11] xstack=inputs=12:layout=0_0|w0_0|0_h0|w0_h0|0_h0+h2|w0_h0+h2|0_h0+h2+h4|w0_h0+h2+h4|0_h0+h2+h4+h6|w0_h0+h2+h4+h6|0_h0+h2+h4+h6+h8|w0_h0+h2+h4+h6+h8[v]" \
-map "[v]" -c:v libx264 -crf 23 -preset veryfast -y /home/dp316/dp316/dc-fang1/outputs/MRIShearingBoxNetFlux/output_grid.mp4
