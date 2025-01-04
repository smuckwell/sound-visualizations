import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Generate example data for demonstration
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Create the figure and set it to fill the display
fig = plt.figure(figsize=(8, 8), dpi=100)
ax = fig.add_subplot(111, projection='3d', position=[0, 0, 1, 1])

# Set background color to white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove axis markers, labels, and ticks
ax.axis('off')

# Plot the wireframe
ax.plot_wireframe(x, y, z, color='blue', linewidth=0.5)

# Adjust the view angle
ax.view_init(elev=30, azim=45)

# Add animation functionality (example of a rotating view)
def update(frame):
    ax.view_init(elev=30, azim=frame)
    return ax,

anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

# Show the plot
plt.show()