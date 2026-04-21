# %%

import numpy as np


def averageTheta(R, Theta, Values):
    """
    R and Theta must be lines.
    I assume log for R and uniform on theta (on the edges at least)
    """
    # dr = np.diff(R)  # log
    # dr_r = dr / R[:-1]
    # dr = np.append(dr, dr_r[-1] * R[-1])
    dtheta = np.diff(Theta)  # lin on the edges
    dtheta = np.append(dtheta, dtheta[-1])

    return np.sum(Values * R[:, np.newaxis] * dtheta[np.newaxis, :], axis=1)

    # return  # rdrdtheta


R = np.logspace(1, 2, 100)
Theta = np.linspace(0, np.pi)
Values = np.ones((len(R), len(Theta)))
print(averageTheta(R, Theta, Values).shape)
# print(0.5 * np.pi * (10**2 - 10**1))

# %%
