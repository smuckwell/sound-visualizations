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
FREQ_LIMIT_HIGH = 5000
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
scale_factor = 0.9
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

# Set the background color and style
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)
ax.grid(True, color='darkgray', alpha=0.3)
ax.set_xlabel('Frequency (Hz)', color='white')
ax.set_ylabel('Amplitude', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

# Initialize the plot with zeros
zeros = np.zeros(n_freqs)
stems = ax.stem(filtered_freqs, zeros)
plt.setp(stems.markerline, 'color', 'cyan', 'markersize', 4)
plt.setp(stems.stemlines, 'color', 'cyan', 'linewidth', 1, 'alpha', 0.7)
plt.setp(stems.baseline, 'color', 'white', 'linewidth', 1)

# Initialize variables for smoothing
last_fft = np.zeros(n_freqs)

def update(frame):
    global last_fft
    
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
        fft_data = np.log1p(fft_data) * 0.3
        
        # Update stem plot data
        stems.markerline.set_ydata(fft_data)
        segments = [np.array([[x, 0], [x, y]]) for x, y in zip(filtered_freqs, fft_data)]
        stems.stemlines.set_segments(segments)
        
        # Adjust y-axis limits dynamically
        max_val = np.max(fft_data)
        ax.set_ylim(0, max_val * 1.1)
        
        return stems.markerline, stems.stemlines, stems.baseline
        
    except Exception as e:
        print(f"Error in update: {e}")
        return stems.markerline, stems.stemlines, stems.baseline

# Create animation
anim = FuncAnimation(
    fig,
    update,
    frames=None,
    interval=10,
    blit=True,
    cache_frame_data=False
)

# Add keyboard event to exit with 'q' or 'Esc'
def on_key(event):
    if event.key in ['q', 'escape']:
        cleanup()
        plt.close('all')

def cleanup():
    stream.stop_stream()
    stream.close()
    audio.terminate()
    plt.close()

fig.canvas.mpl_connect('key_press_event', on_key)

# Adjust layout
plt.tight_layout()

# Show the plot
try:
    plt.show()
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()