import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
import tkinter as tk

# Audio configuration
SAMPLE_RATE = 44100
CHUNK = 1024
FREQ_LIMIT_LOW = 20
FREQ_LIMIT_HIGH = 4000
HISTORY_SIZE = 100
SAVE_COUNT = 100
BACKGROUND_COLOR = 'white'

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

# Get the screen dimensions using tkinter
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate figure dimensions
scale_factor = 1.0
target_dpi = 100
figure_width = (screen_width * scale_factor) / target_dpi
figure_height = (screen_height * scale_factor) / target_dpi

# Set up the plot
plt.rcParams['figure.dpi'] = target_dpi
fig, ax = plt.subplots(figsize=(figure_width, figure_height))
manager = plt.get_current_fig_manager()
if plt.get_backend() == 'TkAgg':
    manager.window.state('zoomed')
elif plt.get_backend() == 'Qt5Agg':
    manager.window.showMaximized()

# Set the background color
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

# Initialize variables
last_fft = np.zeros(n_freqs)
z = np.zeros((HISTORY_SIZE, n_freqs))

# Create meshgrid for streamplot
y, x = np.mgrid[0:HISTORY_SIZE-1:100j, 0:n_freqs-1:100j]

def calculate_vector_field(Z):
    """Calculate U and V components from the Z values"""
    # Calculate gradients
    V = np.gradient(Z, axis=1)
    U = np.gradient(Z, axis=0)
    
    # Interpolate to match the meshgrid size
    from scipy.interpolate import RegularGridInterpolator
    rows, cols = Z.shape
    row_coords = np.arange(rows)
    col_coords = np.arange(cols)
    
    # Create target points for interpolation
    y_new = np.linspace(0, rows-1, 100)
    x_new = np.linspace(0, cols-1, 100)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    points = np.stack((Y_new.ravel(), X_new.ravel()), axis=-1)
    
    # Create interpolators
    U_interp = RegularGridInterpolator((row_coords, col_coords), U)
    V_interp = RegularGridInterpolator((row_coords, col_coords), V)
    
    # Interpolate
    U_mesh = U_interp(points).reshape(100, 100)
    V_mesh = V_interp(points).reshape(100, 100)
    
    return U_mesh, V_mesh

def update(frame):
    global last_fft, z
    
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
        
        # Update z data
        z = np.roll(z, -1, axis=0)
        z[-1, :] = fft_data
        
        # Calculate vector field
        U, V = calculate_vector_field(z)
        
        # Clear previous plot
        ax.clear()
        
        # Create streamplot
        strm = ax.streamplot(x, y, U, V, 
                           density=2,
                           color=np.sqrt(U**2 + V**2),
                           cmap='viridis',
                           linewidth=1,
                           arrowsize=1)
        
        # Customize appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        
    except Exception as e:
        print(f"Error in update: {e}")
        return

# Function to handle cleanup
def cleanup():
    stream.stop_stream()
    stream.close()
    audio.terminate()
    plt.close()

# Add keyboard event to exit with 'q' or 'Esc'
def on_key(event):
    if event.key in ['q', 'escape']:
        cleanup()
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)

# Remove margins
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Create animation
anim = FuncAnimation(
    fig,
    update,
    frames=None,
    interval=20,
    blit=False,
    cache_frame_data=False,
    save_count=SAVE_COUNT
)

# Show the plot
try:
    plt.show()
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()