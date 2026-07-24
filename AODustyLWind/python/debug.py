import matplotlib.pyplot as plt
import numpy as np
from idefix2python import readVTK


v = readVTK("/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs_v3/dw100_v2_thin_tv_b1e4_2000p/vtks/part.0000.vtk")
print(len(v.r))