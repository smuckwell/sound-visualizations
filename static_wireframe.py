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
z_axis_view_angle_rotation = -30

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

# Pre-calculate the meshgrid with wider x range for better screen filling
x = np.linspace(-6, 6, n_freqs)  # Doubled the x range
y = np.linspace(-3, 3, HISTORY_SIZE)
x, y = np.meshgrid(x, y)

# Get the screen dimensions using tkinter
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate oversized figure dimensions
scale_factor = 4.0
target_dpi = 100
figure_width = (screen_width * scale_factor) / target_dpi
figure_height = (screen_height * scale_factor) / target_dpi

# Set the figure size and DPI explicitly
plt.rcParams['figure.dpi'] = target_dpi

# Initialize the plot with full screen
fig = plt.figure(figsize=(figure_width, figure_height))
manager = plt.get_current_fig_manager()
if plt.get_backend() == 'TkAgg':
    manager.window.state('zoomed')  # Works on Windows
elif plt.get_backend() == 'Qt5Agg':
    manager.window.showMaximized()  # Works on Qt
    
# Create subplot that fills the entire figure
ax = fig.add_subplot(111, projection='3d', position=[0, 0, 1, 1])

# Set the background color to white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove axis markers, labels, and ticks
ax.axis('off')

# Initialize variables
last_fft = np.zeros(n_freqs)
z = np.zeros((HISTORY_SIZE, n_freqs))

# Persistent settings for user interactions
persistent_settings = {
    'view_init': (30, z_axis_view_angle_rotation),
    'position': [0, 0, 1, 1],
    'xlim': (-6, 6),  # Wider x limits to match the meshgrid
    'ylim': (-3, 3),
    'zlim': (0, 1),
    'figsize': (figure_width, figure_height),
    'hspace': 0.0,
    'wspace': 0.0,
    'scale': 1.0,
    'aspect': 'auto',
    'offset': [0, 0, 0]  # New: track pan offsets
}

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
        
        # Store current view state before clearing
        current_elev = ax.elev if ax.elev is not None else persistent_settings['view_init'][0]
        current_azim = ax.azim if ax.azim is not None else persistent_settings['view_init'][1]
        
        # Clear the previous wireframe
        ax.clear()
        ax.axis('off')
        
        # Get colors for the wireframe
        segment_colors = get_color_array(z)
        
        # Apply offsets to the data
        x_offset = x + persistent_settings['offset'][0]
        y_offset = y + persistent_settings['offset'][1]
        z_offset = z + persistent_settings['offset'][2]
        
        # Redraw wireframe with colors and thicker lines
        wire = ax.plot_wireframe(x_offset, y_offset, z_offset, 
                               rcount=HISTORY_SIZE, ccount=n_freqs,
                               linewidth=2.0, colors=segment_colors)
        
        # Restore view state
        ax.view_init(current_elev, current_azim)
        
        # Set limits based on current offsets and scale
        ax.set_xlim(persistent_settings['xlim'][0], persistent_settings['xlim'][1])
        ax.set_ylim(persistent_settings['ylim'][0], persistent_settings['ylim'][1])
        ax.set_zlim(persistent_settings['zlim'][0], persistent_settings['zlim'][1])
        
        # Maintain scale and aspect ratio, allowing horizontal stretch
        ax.set_box_aspect(None)  # Allow the plot to stretch horizontally
        ax.set_aspect('auto')
        
    except Exception as e:
        print(f"Error in update: {e}")
        return
    
    return [wire]

# Enhanced event handlers
def on_mouse_press(event):
    if event.inaxes == ax:
        ax._button_pressed = event.button
        ax._mouse_init = (event.xdata, event.ydata)
        ax._offset_init = persistent_settings['offset'].copy()

def on_mouse_release(event):
    ax._button_pressed = None

def on_mouse_motion(event):
    if event.inaxes == ax and ax._button_pressed is not None:
        if ax._button_pressed == 1:  # Left click - rotation (already handled)
            persistent_settings['view_init'] = (ax.elev, ax.azim)
        elif ax._button_pressed == 3:  # Right click - pan
            if ax._mouse_init is not None and event.xdata is not None and event.ydata is not None:
                dx = event.xdata - ax._mouse_init[0]
                dy = event.ydata - ax._mouse_init[1]
                persistent_settings['offset'][0] = ax._offset_init[0] + dx
                persistent_settings['offset'][1] = ax._offset_init[1] + dy
        print_settings()

def on_scroll(event):
    if event.inaxes == ax:
        # Update scale based on scroll direction
        scale_factor = 1.1 if event.button == 'up' else 0.9
        persistent_settings['scale'] *= scale_factor
        # Update limits proportionally
        center_x = sum(persistent_settings['xlim']) / 2
        center_y = sum(persistent_settings['ylim']) / 2
        center_z = sum(persistent_settings['zlim']) / 2
        range_x = persistent_settings['xlim'][1] - persistent_settings['xlim'][0]
        range_y = persistent_settings['ylim'][1] - persistent_settings['ylim'][0]
        range_z = persistent_settings['zlim'][1] - persistent_settings['zlim'][0]
        persistent_settings['xlim'] = (center_x - range_x/2 * scale_factor, 
                                     center_x + range_x/2 * scale_factor)
        persistent_settings['ylim'] = (center_y - range_y/2 * scale_factor,
                                     center_y + range_y/2 * scale_factor)
        persistent_settings['zlim'] = (center_z - range_z/2 * scale_factor,
                                     center_z + range_z/2 * scale_factor)
        print_settings()

# Function to print current settings
def print_settings():
    print("Current settings:")
    print(f"View Init: {persistent_settings['view_init']}")
    print(f"Position: {persistent_settings['position']}")
    print(f"X Lim: {persistent_settings['xlim']}")
    print(f"Y Lim: {persistent_settings['ylim']}")
    print(f"Z Lim: {persistent_settings['zlim']}")
    print(f"Figure Size: {persistent_settings['figsize']}")
    print(f"H Space: {persistent_settings['hspace']}")
    print(f"W Space: {persistent_settings['wspace']}")
    print(f"Scale: {persistent_settings['scale']}")
    print(f"Aspect: {persistent_settings['aspect']}")
    print(f"Offset: {persistent_settings['offset']}")

# Connect enhanced event handlers
fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)
fig.canvas.mpl_connect('scroll_event', on_scroll)

# Remove all margins and spacing
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

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

# Add keyboard event to exit fullscreen with 'q' or 'Esc'
def on_key(event):
    if event.key in ['q', 'escape']:
        cleanup()
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)

# Cleanup function
def cleanup():
    stream.stop_stream()
    stream.close()
    audio.terminate()
    plt.close()

# Keep the animation object in memory and show the plot
try:
    plt.show()
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()