# %%

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 30)
y = np.linspace(0, 1, 40)


def func(x, y):
    return np.array([x**2, y**2])


def func_analytical_div(x, y):
    return np.array(2 * x + 2 * y)


X, Y = np.meshgrid(x, y)
# X, Y = x, y
data = func(X, Y)

# axs = []
# fig = plt.figure()
# axs += [fig.add_subplot(projection="3d")]

analytical_div = func_analytical_div(X, Y)
post_div1 = np.gradient(data[0, :, :], x, axis=1)
post_div2 = np.gradient(data[1, :, :], y, axis=0)
print(np.shape(post_div1))
print(np.shape(post_div2))
post_div = post_div1 + post_div2

fig, axs = plt.subplots(3, figsize=(9, 16))
axs[0].quiver(X, Y, data[0, :, :], data[1, :, :])
pm1 = axs[1].pcolormesh(X, Y, analytical_div)
fig.colorbar(pm1, ax=axs[1])
pm2 = axs[2].pcolormesh(X, Y, analytical_div - post_div)
fig.colorbar(pm2, ax=axs[2])
# fig.colorbar(pm1, ax=axs[1])

for ax in axs:
    ax.set_aspect("equal", adjustable="box")

# %%
