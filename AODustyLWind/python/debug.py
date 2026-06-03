import matplotlib.pyplot as plt
import numpy as np
from idefix2python import readVTK


def get_part_index(v, p):
    r_arr = p.r
    theta_arr = p.theta

    # Calculate absolute physical distance to find the true closest grid cell
    # i = np.array([np.argmin(np.abs(v.r - r)) for r in r_arr])
    i = np.array([np.argmin(np.abs(np.log10(v.r) - np.log10(r))) for r in r_arr])
    j = np.array([np.argmin(np.abs(v.theta - th)) for th in theta_arr])

    return j, i


v_sample = readVTK(
    "/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs/dw100_v2_b3e3_2000p/vtks/data.0000.vtk"
)
print(f"Grid shape: {v_sample.data['RHO'].shape}")
print(f"Expected: (N_theta: {len(v_sample.theta)}, N_r: {len(v_sample.r)})")

tstops_g = []
tstops = []
for fr in range(0, 100, 5):
    v = readVTK(
        f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs/dw100_v2_b3e3_2000p/vtks/data.{fr:04}.vtk"
    )
    p = readVTK(
        f"/home/dp316/dp316/dc-fang1/IdefixRuns/AODustyLWind/outputs/dw100_v2_b3e3_2000p/vtks/part.{fr:04}.vtk"
    )

    # R = p.r * np.sin(p.theta)
    tstops_g += [p.data["TSTOP"][0]]

    j, i = get_part_index(v, p)
    rho = v.data["RHO"][i[0], j[0], 0]
    prs = v.data["PRS"][i[0], j[0], 0]
    cs = np.sqrt(prs / rho)
    print(f"fr={fr} | Gas Time (v.t)={v.t[0]:.4f} | Part Time (p.t)={p.t[0]:.4f}")
    print(f"  TSTOP (C++)    = {p.data['TSTOP'][0]:.4e}")
    print(f"  DRAGCOEFF (Py) = {p.data['DRAGCOEFF'][0]:.4e}")
    print(f"  rho (Py)       = {rho:.4e}")
    print(f"  cs (Py)        = {cs:.4e}")
    print("-" * 40)
    tstop = p.data["DRAGCOEFF"][0] / (rho * cs)
    tstops += [tstop]

fig, ax = plt.subplots(2)

ax[0].plot(tstops, label="not good")
ax[0].plot(tstops_g, label="good", ls="--")
ax[0].legend()
ax[1].plot(np.asarray(tstops) / np.asarray(tstops_g))
ax[0].set_ylim(0, 1)
fig.savefig("debug.png")
print("debug.png")
