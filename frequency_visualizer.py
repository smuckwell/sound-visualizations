import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft
from collections import deque
import time

# Parameters
SAMPLE_RATE = 44100  # Sample rate in Hz
BUFFER_SIZE = 1024  # Number of frames per buffer
HISTORY_LENGTH = 100  # Number of past frequency samples to keep

# Setting up a deque to store frequency history
frequency_history = deque(maxlen=HISTORY_LENGTH)

# Setting up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update_plot(magnitude):
    frequency_history.append(magnitude)
    z_data = np.array(frequency_history).T  # Transpose to match the shape
    x_data, y_data = np.meshgrid(np.arange(z_data.shape[1]), np.arange(z_data.shape[0]))
    ax.clear()
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency Bin')
    ax.set_zlabel('Amplitude')
    ax.plot_surface(x_data, y_data, z_data, cmap=cm.viridis)
    plt.draw()
    plt.pause(0.01)

# Ensure default_speaker is correctly instantiated
default_speaker = sd.InputStream(samplerate=SAMPLE_RATE, channels=1)

# Capturing audio output
def capture_audio():
    with default_speaker:
        while True:
            data, overflowed = default_speaker.read(BUFFER_SIZE)
            if overflowed:
                print("Buffer overflowed")
            # Apply FFT to get frequency data
            fft_data = fft(data[:, 0])  # Use the first channel if stereo
            magnitude = np.abs(fft_data[:BUFFER_SIZE // 2])  # Only use the positive frequencies
            # Update the plot with the current frequency data
            update_plot(magnitude)
            time.sleep(0.01)  # Small delay to allow real-time plotting

# Running the audio capture and visualization
if __name__ == "__main__":
    plt.ion()  # Turn on interactive mode for real-time updates
    capture_audio()
