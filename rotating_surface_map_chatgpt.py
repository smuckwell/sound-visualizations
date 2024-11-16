import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from mpl_toolkits.mplot3d import Axes3D

# Seed prompt
# I need help writing some software using Python.
# I would like to capture the audio picked up by the microphone of this device. I would like to display the spectrum captured as it changes over time so it probably needs to be saved to some kind of internal array. The displayed plot should be a surface, using the turbo color map. I would like the displayed frequencies to be limited, between 20 hz and 5000 hz. I would like the plot to continuously rotate smoothly about the Z axis, with the rotation speed changing corresponding to the current dominant frequency. I would like the animation of the surface changing and rotation to be very smooth, so use techniques that can improve the display characteristics.

# Audio configuration
SAMPLE_RATE = 44100  # 44.1 kHz
CHUNK = 1024  # Number of audio samples per frame
FREQ_LIMIT_LOW = 20  # Frequency limit for the plot (Hz)
FREQ_LIMIT_HIGH = 5000  # Frequency limit for the plot (Hz)

# PyAudio initialization
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

# Set up figure and 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, 1, CHUNK // 2)
y = np.linspace(0, 1, 100)  # We'll maintain history of 100 frames
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)
surface = ax.plot_surface(x, y, z, cmap='turbo')

# Set axis labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('Frequency')
ax.set_ylabel('Time')
ax.set_zlabel('Amplitude')

rotation_speed = 1.0  # Initial rotation speed

# Update function for animation
def update(frame):
    global rotation_speed
    # Read data from microphone
    data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
    # Perform FFT
    fft_data = np.abs(fft(data))[:CHUNK // 2]
    freqs = fftfreq(CHUNK, 1 / SAMPLE_RATE)[:CHUNK // 2]

    # Limit frequencies to range of interest
    mask = (freqs >= FREQ_LIMIT_LOW) & (freqs <= FREQ_LIMIT_HIGH)
    fft_data = fft_data[mask]
    freqs = freqs[mask]

    # Update Z data for the surface plot
    z[:-1, :] = z[1:, :]  # Shift old data
    z[-1, :len(fft_data)] = fft_data / np.max(fft_data) if np.max(fft_data) > 0 else 0  # Add new data
    ax.clear()

    # Redraw surface
    ax.plot_surface(x, y, z, cmap='turbo')

    # Calculate dominant frequency and update rotation speed
    dominant_freq = freqs[np.argmax(fft_data)] if len(freqs) > 0 else 0
    rotation_speed = dominant_freq / 100.0  # Scale rotation speed with dominant frequency

    # Rotate the plot
    ax.view_init(30, frame * rotation_speed)

# Animate the plot
ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 100), interval=50, blit=False)
plt.show()

# Stop audio stream on exit
stream.stop_stream()
stream.close()
audio.terminate()
