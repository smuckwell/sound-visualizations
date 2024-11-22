import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import tkinter as tk

# Audio configuration remains the same
SAMPLE_RATE = 44100
CHUNK = 1024
FREQ_LIMIT_LOW = 1
FREQ_LIMIT_HIGH = 5000
HISTORY_SIZE = 100
SAVE_COUNT = 100

z_axis_scaling = 0.5
z_axis_view_angle_rotation = -30  # Adjusted view angle for better centering

# PyAudio initialization
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=None
)

# Calculate frequency mask once
freqs = fftfreq(CHUNK, 1 / SAMPLE_RATE)[:CHUNK // 2]
freq_mask = (freqs >= FREQ_LIMIT_LOW) & (freqs <= FREQ_LIMIT_HIGH)
filtered_freqs = freqs[freq_mask]
n_freqs = len(filtered_freqs)

# Pre-calculate the meshgrid with adjusted range
x = np.linspace(-3, 3, n_freqs)    # Centered range
y = np.linspace(-3, 3, HISTORY_SIZE)  # Centered range
x, y = np.meshgrid(x, y)

# Get the screen dimensions using tkinter
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Calculate oversized figure dimensions
scale_factor = 4.0
target_dpi = 100
figure_width = (screen_width * scale_factor) / target_dpi
figure_height = (screen_height * scale_factor) / target_dpi

# Set the figure size and DPI explicitly
plt.rcParams['figure.dpi'] = target_dpi
plt.rcParams['figure.figsize'] = [figure_width, figure_height]

# Create the oversized figure
fig = plt.figure(
    figsize=(figure_width, figure_height),
    dpi=target_dpi,
    facecolor='white'
)

# Make figure fullscreen
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')  # For Windows
# mng.full_screen_toggle()  # For Linux

# Create 3D axes with extended position
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

# Set axes to extend beyond figure boundaries with adjusted centering
# Format: [left, bottom, width, height]
ax.set_position([-10.0, -1.0, 10.0, 4.0]) # Adjusted left position for centering

# Set the background colors
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Initialize the z data
z = np.zeros((HISTORY_SIZE, n_freqs))
wire = None

# Set up the plot with centered limits
ax.set_xlim(-3, 3)  # Centered limits
ax.set_ylim(-3, 3)  # Centered limits
ax.set_zlim(0, 4)   
ax.axis('off')

# Set view angle
ax.view_init(30, z_axis_view_angle_rotation)

# Initialize variables
last_fft = np.zeros(n_freqs)

def get_color_array(z_values):
    z_min, z_max = np.min(z_values), np.max(z_values)
    if z_min == z_max:
        normalized = np.zeros_like(z_values)
    else:
        normalized = (z_values - z_min) / (z_max - z_min)
    colors = plt.cm.turbo(normalized)
    colors_reshaped = colors.reshape(-1, 4)
    return colors_reshaped

def update(frame):
    global wire, last_fft, z
    
    try:
        # Read audio data
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        
        # Compute FFT and apply frequency mask
        fft_data = np.abs(fft(data))[:CHUNK // 2]
        fft_data = fft_data[freq_mask]
        
        # Apply smoothing
        smoothing_factor = 0.7
        fft_data = smoothing_factor * last_fft + (1 - smoothing_factor) * fft_data
        last_fft = fft_data.copy()
        
        # Normalize
        fft_max = np.max(fft_data)
        if fft_max > 0:
            fft_data = fft_data / fft_max
        
        # Apply logarithmic transformation with increased scaling
        fft_data = np.log1p(fft_data) * z_axis_scaling * 4
        
        # Update z data
        z = np.roll(z, -1, axis=0)
        z[-1, :] = fft_data
        
        # Clear the previous wireframe
        ax.clear()
        ax.axis('off')
        
        # Get colors for the wireframe
        segment_colors = get_color_array(z)
        
        # Redraw wireframe with colors and thicker lines
        wire = ax.plot_wireframe(x, y, z, rcount=HISTORY_SIZE, ccount=n_freqs,
                               linewidth=2.0, colors=segment_colors)
        
        # Maintain view angle
        ax.view_init(30, z_axis_view_angle_rotation)
        
        # Reset the extended limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(0, np.max(z) * 2)
        
        # Maintain extended position
        ax.set_position([-2.0, -1.0, 6.0, 4.0])
        
    except Exception as e:
        print(f"Error in update: {e}")
        return
    
    return [wire]

# Create animation
anim = FuncAnimation(
    fig,
    update,
    frames=None,
    interval=50,
    blit=False,
    cache_frame_data=False,
    save_count=SAVE_COUNT
)

# Cleanup function
def cleanup():
    stream.stop_stream()
    stream.close()
    audio.terminate()
    plt.close()

# Add keyboard event to exit fullscreen with 'q' or 'Esc'
def on_key(event):
    if event.key in ['q', 'escape']:
        cleanup()
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)

# Keep the animation object in memory and show the plot
try:
    plt.show()
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()