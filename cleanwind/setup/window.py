import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True})
epsilon = 0.05
epsilonTop = 0.2
Hideal = 5.0
zbound = 50
Z = np.linspace(-zbound, zbound, 1000)

smooth = 0.2
Rin = 1.0


def window(rc, z):
    R0 = np.clip(rc, Rin, None)
    return 0.5 * (1 - np.tanh((abs(z) - Hideal * epsilon * R0) / smooth))


def window2(rc, z):
    R0 = np.clip(rc, Rin, None)
    zh = z / (R0 * epsilon)  # z in units of disc scale height h=R*epsilon

    return 0.5 * (1 - np.tanh((abs(zh) - Hideal) / smooth))


fig, axs = plt.subplots(1, 3, squeeze=False)
rc = 50.0

ax = axs[0, 0]
ax.plot(Z, window(rc, Z), label="1", alpha=0.5)
ax.plot(Z, window2(rc, Z), label="2", alpha=0.5)


ax.set_ylim(0, 1.1)
ax.legend()
ax.set_xlabel("$z$")

ax = axs[0, 1]
R = np.linspace(0, rc)
Rgrid, Zgrid = np.meshgrid(R, Z)
ax.pcolormesh(R, Z, window(Rgrid, Zgrid))

ax = axs[0, 2]
ax.pcolormesh(R, Z, window2(Rgrid, Zgrid))

for ax in axs[1:]:
    ax.set_aspect("equal")

fig.savefig("fig.png", dpi=400)
print("fig.png")
