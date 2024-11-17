import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from mpl_toolkits.mplot3d import Axes3D

# Audio configuration
SAMPLE_RATE = 44100
CHUNK = 1024
FREQ_LIMIT_LOW = 1
FREQ_LIMIT_HIGH = 5000
HISTORY_SIZE = 100
SAVE_COUNT = 100  # Number of frames to keep in memory

z_axis_scaling = 0.5
z_axis_rotation_speed = 3.0

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

# Initialize the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the background color to black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Initialize the z data
z = np.zeros((HISTORY_SIZE, n_freqs))

# Create initial surface plot
surf = ax.plot_surface(x, y, z, cmap='turbo')

# Set up the plot
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.axis('off')  # Remove axis markers, labels, and ticks

# Initialize variables
rotation_angle = 0
rotation_speed = z_axis_rotation_speed
last_fft = np.zeros(n_freqs)

def update(frame):
    global rotation_angle, rotation_speed, surf, last_fft, z
    
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
        fft_data = np.log1p(fft_data) * z_axis_scaling  # log1p is used to avoid log(0)
        
        # Update z data
        z = np.roll(z, -1, axis=0)
        z[-1, :] = fft_data
        
        # Update rotation
        dominant_freq_idx = np.argmax(fft_data)
        dominant_freq = filtered_freqs[dominant_freq_idx] if fft_max > 0 else 0
        target_speed = np.clip(dominant_freq / 1000.0, 0.5, 5.0)
        rotation_speed = 0.95 * rotation_speed + 0.05 * target_speed
        rotation_angle = (rotation_angle + rotation_speed) % 360
        
        # Clear the previous surface
        ax.clear()
        
        # Redraw surface and set view
        surf = ax.plot_surface(x, y, z, cmap='turbo', antialiased=False)
        ax.view_init(30, rotation_angle)
        
        # Reset the limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, np.max(z))
        ax.axis('off')  # Remove axis markers, labels, and ticks
        
    except Exception as e:
        print(f"Error in update: {e}")
        return
    
    return [surf]

# Create animation with explicit save_count
anim = FuncAnimation(
    fig,
    update,
    frames=None,
    interval=50,
    blit=False,
    cache_frame_data=False,  # Disable frame caching
    save_count=SAVE_COUNT    # Explicit save count
)

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