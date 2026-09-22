import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True})
epsilon = 0.05


def density_profile(r, theta):
    rc = r * np.sin(theta)
    Rmin = max(rc, 1.0)
    z = r * np.cos(theta)
    cs2 = 0.05**2 / Rmin
    # return Rmin ** (-1.5) * np.exp(1.0 / cs2 * (1.0 / r - 1.0 / Rmin))
    H = epsilon * Rmin
    return Rmin ** (-1.5) * np.exp(-(z**2) / (2 * H**2))


R = np.linspace(1, 1.05, 1000)
Theta = np.linspace(0, np.pi, 500)
Rgrid, Thetagrid = np.meshgrid(R, Theta)
X, Z = Rgrid * np.sin(Thetagrid), Rgrid * np.cos(Thetagrid)

fig, axes = plt.subplots(2, squeeze=False, figsize=(8, 6))
mesh = np.zeros(Rgrid.shape)
print(mesh.shape)
for i, r in enumerate(R):
    for j, theta in enumerate(Theta):
        mesh[j, i] = density_profile(r, theta)

cmesh = axes[0, 0].pcolormesh(X, Z, mesh)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].set_aspect("equal")
fig.colorbar(cmesh, ax=axes[0, 0])

line = []
theta0 = np.pi / 3
for r in R:
    line.append(density_profile(r, theta0))
axes[1, 0].plot(R, line, "-x")
axes[1, 0].set_xlim(0, 1.1)

fig.savefig("fig.png", dpi=400)
print("fig.png")
