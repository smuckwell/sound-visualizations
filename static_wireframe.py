import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

# Audio configuration
SAMPLE_RATE = 44100
CHUNK = 1024
FREQ_LIMIT_LOW = 1
FREQ_LIMIT_HIGH = 5000
HISTORY_SIZE = 100
SAVE_COUNT = 100  # Number of frames to keep in memory

z_axis_scaling = 0.5

z_axis_view_angle_rotation = 15

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

# Pre-calculate the meshgrid
x = np.linspace(0, 1, n_freqs)
y = np.linspace(0, 1, HISTORY_SIZE)
x, y = np.meshgrid(x, y)

# Initialize the plot in fullscreen
plt.rcParams['figure.figsize'] = [plt.get_current_fig_manager().window.winfo_screenwidth()/100, 
                                 plt.get_current_fig_manager().window.winfo_screenheight()/100]
fig = plt.figure(figsize=(10, 8))
fig.canvas.manager.window.attributes('-fullscreen', False)

# Create 3D axes with a specific position for top left
ax = fig.add_subplot(111, projection='3d')

# Center the plot within the screen
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Adjust these margins to move the plot to top left
left_margin = -0.4   # Reduce left margin to move left
top_margin = -0.4    # Reduce top margin to move up
width = 1.3         # Adjust width of plot
height = 1.2        # Adjust height of plot
ax.set_position([left_margin, 1-height-top_margin, width, height])

# Set the background color to white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Initialize the z data
z = np.zeros((HISTORY_SIZE, n_freqs))

# Create initial wireframe plot
wire = None

# Set up the plot
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.axis('off')  # Remove axis markers, labels, and ticks

# Set static view angle for top-left orientation
# ax.view_init(20, 240)  # Modified view angle
ax.view_init(20, z_axis_view_angle_rotation)  # 

# Initialize variables
last_fft = np.zeros(n_freqs)

def get_color_array(z_values):
    # Normalize z values to 0-1 range
    z_min, z_max = np.min(z_values), np.max(z_values)
    if z_min == z_max:
        normalized = np.zeros_like(z_values)
    else:
        normalized = (z_values - z_min) / (z_max - z_min)
    
    # Get colors from turbo colormap
    colors = plt.cm.turbo(normalized)
    
    # Convert to the format expected by plot_wireframe
    colors_reshaped = colors.reshape(-1, 4)  # Reshape to (N, 4) array
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
        
        # Apply logarithmic transformation
        fft_data = np.log1p(fft_data) * z_axis_scaling
        
        # Update z data
        z = np.roll(z, -1, axis=0)
        z[-1, :] = fft_data
        
        # Clear the previous wireframe
        ax.clear()
        ax.axis('off')
        
        # Get colors for the wireframe
        segment_colors = get_color_array(z)
        
        # Redraw wireframe with colors
        wire = ax.plot_wireframe(x, y, z, rcount=HISTORY_SIZE, ccount=n_freqs,
                               linewidth=0.5, colors=segment_colors)
        
        # Maintain view angle
        ax.view_init(20, z_axis_view_angle_rotation)
        
        # Reset the limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, np.max(z))
        
        # Maintain the top-left position
        ax.set_position([left_margin, 1-height-top_margin, width, height])
        
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