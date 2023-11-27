import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # Or try 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WXAgg'
data_plot = np.load("data_plot.npy")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_title('$[x_{R1V}, x_{R2L}, x_{R2T}, x_{R2V}, x_{R3V}]_t$ in PC Space')
ax.set_xlim(data_plot[:, 0].min() - 1, data_plot[:, 0].max() + 1)
ax.set_ylim(data_plot[:, 1].min() - 1, data_plot[:, 1].max() + 1)
text = ax.text(6, -6, "Sample Text", fontsize=12, color='red')

# Store scatter points
scatter_points = []

# Animation function for showing scatter points one by one
def update(num):
    point = ax.scatter(data_plot[num, 0], data_plot[num, 1], color='blue' if num < healthy_data.shape[0] else 'red', marker = 'o', s=50)
    scatter_points.append(point)
    return scatter_points

# Creating the new animation
ani = animation.FuncAnimation(fig, update, frames=len(data_plot), interval=10, blit=True)

ani.save('animated_scatter_plot_points.gif', writer='pillow')

plt.close()  # Close the plot to finalize the animation file creation