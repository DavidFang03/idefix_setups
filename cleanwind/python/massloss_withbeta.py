import matplotlib.pyplot as plt
import numpy as np

betas_float = [1e3, 4e3, 1e4, 6e4, 1e5]
zetasp = [1.5e-3, 1.5e-4, 3.5e-5, 4.5e-6, 3e-6]

a, b = np.polyfit(np.log(betas_float), np.log(zetasp), deg=1)
a2, b2 = np.polyfit(np.log(betas_float[-2:]), np.log(zetasp[-2:]), deg=1)
expb = np.exp(b)
expb2 = np.exp(b2)

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

ax.plot(betas_float, zetasp, "x")
ax.plot(betas_float, betas_float**a * expb, label=rf"${expb:.2f}\beta^{{{a:.2f}}}$")
ax.plot(betas_float, betas_float**a2 * expb2, label=rf"${expb2:.2f}\beta^{{{a2:.2f}}}$")

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\zeta(R=50\,\mathrm{au})$")
ax.legend()
fig.suptitle("Scaling of the mass loss rate with the (midplane) plasma beta")
path = "/home/dp316/dp316/dc-fang1/IdefixRuns/cleanwind/python/massloss.png"
fig.savefig(path, dpi=200)
print(f"[OK] {path}")
