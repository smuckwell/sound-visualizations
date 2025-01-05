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
FREQ_LIMIT_HIGH = 8000
HISTORY_SIZE = 100
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
scale_factor = 0.8
target_dpi = 100
figure_width = (screen_width * scale_factor) / target_dpi
figure_height = (screen_height * scale_factor) / target_dpi

# Set up the figure
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
frequency_history = []
amplitude_history = []

def update(frame):
    global last_fft, frequency_history, amplitude_history
    
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
        
        # Store data points
        for freq, amp in zip(filtered_freqs, fft_data):
            frequency_history.append(freq)
            amplitude_history.append(amp)
        
        # Keep only recent history
        max_points = HISTORY_SIZE * n_freqs
        if len(frequency_history) > max_points:
            frequency_history = frequency_history[-max_points:]
            amplitude_history = amplitude_history[-max_points:]
        
        # Clear previous plot
        ax.clear()
        
        # Create hexbin plot
        hb = ax.hexbin(
            frequency_history,
            amplitude_history,
            gridsize=50,
            cmap='viridis',
            bins='log',
            extent=[FREQ_LIMIT_LOW, FREQ_LIMIT_HIGH, 0, 1]
        )
        
        # Customize appearance
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.set_xlim(FREQ_LIMIT_LOW, FREQ_LIMIT_HIGH)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Frequency (Hz)', color='white')
        ax.set_ylabel('Amplitude', color='white')
        ax.tick_params(colors='white')
        
        # Add colorbar
        if not hasattr(fig, 'colorbar'):
            fig.colorbar = plt.colorbar(hb)
        fig.colorbar.ax.tick_params(colors='white')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        
    except Exception as e:
        print(f"Error in update: {e}")
        return

# Add keyboard event to exit with 'q' or 'Esc'
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

# Create animation
anim = FuncAnimation(
    fig,
    update,
    frames=None,
    interval=10,
    blit=False,
    cache_frame_data=False
)

# Remove margins
plt.tight_layout()

# Keep the animation object in memory and show the plot
try:
    plt.show()
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()