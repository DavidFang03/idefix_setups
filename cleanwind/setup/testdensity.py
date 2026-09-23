import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.rcParams.update({"text.usetex": True})
epsilon = 0.05
epsilonTop = 0.2
Hideal = 5.0


def density_profile(r, theta):
    rc = r * np.sin(theta)
    Rmin = max(rc, 1.0)
    z = r * np.cos(theta)
    cs2 = 0.05**2 / Rmin
    # return Rmin ** (-1.5) * np.exp(1.0 / cs2 * (1.0 / r - 1.0 / Rmin))
    H = epsilon * Rmin
    return Rmin ** (-1.5) * np.exp(-(z**2) / (2 * H**2))


def temperature_profile(r, theta):
    Rin = 1.0
    trSmoothingTemp = 1.0
    z = r * np.cos(theta)
    R = r * np.sin(theta)
    R0 = max(R, Rin)
    # return epsilon**2 / r
    Zh = abs(z / R0) / epsilon
    Tdisk = epsilon * epsilon / r
    Tcorona = epsilonTop * epsilonTop / r
    # Tdisk = epsilon * epsilon / R0
    # Tcorona = epsilonTop * epsilonTop / R0
    return 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * np.tanh(
        abs(z / Rin) - Hideal * epsilon * R0 / Rin
    )
    # return 0.5 * (Tdisk + Tcorona) + 0.5 * (Tcorona - Tdisk) * np.tanh(
    #     (Zh - Hideal) / trSmoothingTemp
    # )


def window(rc, z):
    Rin = 1.0
    R0 = max(rc, Rin)
    return np.tanh(abs(z / Rin) - Hideal * epsilon * R0 / Rin)


R = np.linspace(1, 5, 1000)
Theta = np.linspace(0, np.pi, 500)
Rgrid, Thetagrid = np.meshgrid(R, Theta)
X, Z = Rgrid * np.sin(Thetagrid), Rgrid * np.cos(Thetagrid)

fig, axes = plt.subplots(2, 3, squeeze=False, figsize=(8, 6))
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


## temperature
mesh = np.zeros(Rgrid.shape)
print(mesh.shape)
for i, r in enumerate(R):
    for j, theta in enumerate(Theta):
        mesh[j, i] = temperature_profile(r, theta)


ax1 = axes[0, 1]
cmesh = ax1.pcolormesh(X, Z, mesh, norm=matplotlib.colors.LogNorm())
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_aspect("equal")
fig.colorbar(cmesh, ax=ax1)

line = []
theta0 = np.pi / 6
for r in R:
    line.append(temperature_profile(r, theta0))
axes[1, 1].plot(R, line)

Z = np.linspace(-2, 2)

for rc in [0.1, 0.5, 1, 1.5]:
    axes[1, 2].plot(Z, window(rc, Z), label=rc, alpha=0.5)
axes[1, 2].legend()

fig.savefig("fig.png", dpi=400)
print("fig.png")
