# %%
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Define a 1D grid
x = np.logspace(0, 1, 11)  # x-coordinates (0, 1, 2, ..., 10)
# x = np.linspace(0, 10, 11)  # x-coordinates (0, 1, 2, ..., 10)
print(x)

# Define values on this grid (e.g., a function of x)
values = np.sin(x)  # Sample values (sine function)

# Create a RegularGridInterpolator object
interp_func = RegularGridInterpolator((x,), values)

# Points to interpolate at (these can be any value within the range of x)
xi = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])

# Perform interpolation
interpolated_values = interp_func(xi)

# Print interpolated values
print("Interpolated values:", interpolated_values)

# Visualization
plt.plot(x, values, "o", label="Data points (sin(x))")
plt.plot(xi, interpolated_values, "x", label="Interpolated values", color="red")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("1D Interpolation using RegularGridInterpolator")
plt.legend()
plt.grid()
plt.show()
# %%
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Define 2D grid points
x = np.logspace(0, 1, 15)  # x-coordinates (0, 1, 2, 3, 4, 5)
y = np.logspace(0, 1, 15)  # y-coordinates (0, 1, 2, 3, 4, 5)
print(x, y)

# Define custom values on the grid (a 2D function)
# For example, a simple function: z = f(x, y) = sin(sqrt(x^2 + y^2))
X, Y = np.meshgrid(x, y)  # Create a meshgrid for x and y
Z = np.sin(np.sqrt(X**2 + Y**2))  # Compute custom values

# Create the RegularGridInterpolator object
interp_func = RegularGridInterpolator((x, y), Z)

# Define points where we want to interpolate
xi = np.array([[1.5, 1.5], [2.5, 2.5], [3.5, 3.5], [4.2, 1.8]])

# Perform interpolation
interpolated_values = interp_func(xi)

# Print interpolated values
print("Interpolated values at specified points:", interpolated_values)

# Visualization
# Create a grid for visualization
xi_grid, yi_grid = np.meshgrid(np.linspace(1, 10, 5), np.linspace(1, 10, 5))
zi_grid = interp_func(np.array([xi_grid.ravel(), yi_grid.ravel()]).T).reshape(
    xi_grid.shape
)
print(zi_grid)

plt.figure(figsize=(10, 6))
plt.contourf(xi_grid, yi_grid, zi_grid, levels=50, cmap="viridis")
plt.colorbar(label="Interpolated Values")
plt.scatter(xi[:, 0], xi[:, 1], color="red", label="Interpolation Points", zorder=5)
plt.title("2D Interpolation using RegularGridInterpolator with Custom Values")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()


def get_streamplot_data(
    Xpoints,
    Ypoints,
    Ux_data,
    resolution=200,
    method="linear",
    # Xpoints, Ypoints, Ux_data, Uy_data, resolution=200, method="linear"
):
    """
    X, Y doivent provenir d'un meshgrid. Ux et Uy sont les points de données.
    Renvoie les tableaux 1D x_coords et y_coords, avec les valeurs interpolées
    Ux_uni, Uy_uni (tableaux 2D) prêts pour plt.streamplot.
    """
    # 1. Extraire les coordonnées 1D
    x_1d = Xpoints[0, :]
    y_1d = Ypoints[:, 0]

    # 2. S'assurer qu'elles sont strictement croissantes (Requis par RegularGridInterpolator)
    idx_x = np.argsort(x_1d)
    idx_y = np.argsort(y_1d)

    x_1d_sorted = x_1d[idx_x]
    y_1d_sorted = y_1d[idx_y]

    # 3. Réordonner les données de vitesse en fonction des index triés
    Ux_sorted = Ux_data[np.ix_(idx_y, idx_x)]
    # Uy_sorted = Uy_data[np.ix_(idx_y, idx_x)]

    # 5. Créer les interpolateurs
    # On passe la transposée (.T) pour que les axes correspondent à (x, y)
    Ux_interp = RegularGridInterpolator(
        (x_1d_sorted, y_1d_sorted),
        Ux_sorted.T,
        fill_value=0,
        method=method,
        bounds_error=False,
    )
    # Uy_interp = RegularGridInterpolator(
    #     (x_1d_sorted, y_1d_sorted),
    #     Uy_sorted.T,
    #     fill_value=0,
    #     method=method,
    #     bounds_error=False,
    # )

    # 4. Créer la nouvelle grille régulière
    x_min, x_max = x_1d_sorted[0], x_1d_sorted[-1]
    y_min, y_max = y_1d_sorted[0], y_1d_sorted[-1]

    x_coords = x_min + np.arange(resolution) * ((x_max - x_min) / (resolution - 1))
    y_coords = y_min + np.arange(resolution) * ((y_max - y_min) / (resolution - 1))

    X_uni, Y_uni = np.meshgrid(x_coords, y_coords)

    # 6. Évaluer les interpolateurs en empilant les points proprement
    pts = np.stack((X_uni, Y_uni), axis=-1)

    return (
        x_coords,
        y_coords,
        Ux_interp(pts),
        # Uy_interp(pts),
    )


xgrid, ygrid, interpoints = get_streamplot_data(X, Y, Z, resolution=5)
print(interpoints)

# %%
