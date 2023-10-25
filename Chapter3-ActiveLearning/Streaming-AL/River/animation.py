import numpy as np
import matplotlib.pyplot as plt

# Define Gaussian function
def gaussian(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a domain for our Gaussian
x = np.linspace(-10, 10, 400)
y = gaussian(x, 0, 2)
Y, X = np.meshgrid(np.linspace(0, 2, 10), x)  # for a ribbon along the y-axis
Z = np.tile(y, (10, 1)).T

# Plot the surface (ribbon)
ax.plot_surface(X, Y, Z, alpha=0.5, color='blue')

# Set the axis labels and title

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.axis('off')

plt.show()
