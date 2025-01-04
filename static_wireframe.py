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
root.withdraw()  # Hide the root window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate oversized figure dimensions
scale_factor = 4.0
target_dpi = 100
figure_width = (screen_width * scale_factor) / target_dpi
figure_height = (screen_height * scale_factor) / target_dpi

# Set the figure size and DPI explicitly
plt.rcParams['figure.dpi'] = target_dpi

# Initialize the plot
fig = plt.figure(figsize=(figure_width, figure_height))
ax = fig.add_subplot(111, projection='3d')

# Maximize the figure window
plt.get_current_fig_manager().window.state('zoomed')

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
    'position': [0, 0, 1, 1],  # Fill the entire figure
    'xlim': (-3, 3),
    'ylim': (-3, 3),
    'zlim': (0, 1),
    'figsize': (figure_width, figure_height),
    'hspace': 0.0,
    'wspace': 0.0,
    'scale': 1.0,
    'aspect': 'auto'
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
        
        # Clear the previous wireframe
        ax.clear()
        ax.axis('off')
        
        # Get colors for the wireframe
        segment_colors = get_color_array(z)
        
        # Redraw wireframe with colors and thicker lines
        wire = ax.plot_wireframe(x, y, z, rcount=HISTORY_SIZE, ccount=n_freqs,
                               linewidth=2.0, colors=segment_colors)
        
        # Maintain view angle
        ax.view_init(*persistent_settings['view_init'])
        
        # Reset the extended limits
        ax.set_xlim(*persistent_settings['xlim'])
        ax.set_ylim(*persistent_settings['ylim'])
        ax.set_zlim(*persistent_settings['zlim'])
        
        # Maintain extended position
        ax.set_position(persistent_settings['position'])
        
        # Update figure size
        fig.set_size_inches(*persistent_settings['figsize'])
        
        # Update subplot spacing
        fig.subplots_adjust(hspace=persistent_settings['hspace'], wspace=persistent_settings['wspace'])
        
        # Update scale and aspect ratio
        ax.set_box_aspect([1, 1, persistent_settings['scale']])
        ax.set_aspect(persistent_settings['aspect'])
        
    except Exception as e:
        print(f"Error in update: {e}")
        return
    
    return [wire]

# Function to capture user interactions
def on_resize(event):
    persistent_settings['position'] = ax.get_position().bounds
    persistent_settings['figsize'] = fig.get_size_inches()
    persistent_settings['hspace'] = fig.subplotpars.hspace
    persistent_settings['wspace'] = fig.subplotpars.wspace
    print_settings()

def on_rotate(event):
    persistent_settings['view_init'] = ax.elev, ax.azim
    print_settings()

def on_xlim_changed(event):
    persistent_settings['xlim'] = ax.get_xlim()
    print_settings()

def on_ylim_changed(event):
    persistent_settings['ylim'] = ax.get_ylim()
    print_settings()

def on_zlim_changed(event):
    persistent_settings['zlim'] = ax.get_zlim()
    print_settings()

def on_draw(event):
    persistent_settings['scale'] = ax.get_box_aspect()[2]
    persistent_settings['aspect'] = ax.get_aspect()
    #print_settings()

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

# Connect event handlers
fig.canvas.mpl_connect('resize_event', on_resize)
fig.canvas.mpl_connect('motion_notify_event', on_rotate)
fig.canvas.mpl_connect('draw_event', on_draw)
ax.callbacks.connect('xlim_changed', on_xlim_changed)
ax.callbacks.connect('ylim_changed', on_ylim_changed)
ax.callbacks.connect('zlim_changed', on_zlim_changed)

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