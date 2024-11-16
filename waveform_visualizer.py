import soundcard as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Affine2D

# Get the default microphone with loopback enabled
mic = sc.default_microphone()
loopback = mic.recorder(samplerate=48000, channels=2, blocksize=4096)  # Increase blocksize

# Set parameters
sample_rate = 48000  # Hz
channels = 2
min_freq = 20  # Minimum audible frequency in Hz
max_freq = 200  # Maximum audible frequency in Hz is 20,000 Hz. Narrow the range for better visualization.

visualization_title = 'Apres Ski Waveforms'

# Create a colormap that spans from violet to red
cmap = plt.get_cmap('turbo')
norm = Normalize(vmin=min_freq, vmax=max_freq)
scalar_map = ScalarMappable(norm=norm, cmap=cmap)

# Function to update the waveform in real-time
def update_waveform(frame, line, loopback, text, counter):
    audio_data = loopback.record(numframes=4096)  # Match blocksize
    audio_data_mono = audio_data[:, 0] if channels > 1 else audio_data

    # Calculate the FFT to find the dominant frequency
    fft_data = np.fft.fft(audio_data_mono)
    freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft_data = np.abs(fft_data[:len(fft_data)//2])
    dominant_freq = positive_freqs[np.argmax(positive_fft_data)]

    # Map the dominant frequency to a color in the colormap
    color = scalar_map.to_rgba(dominant_freq)

    # Change the color of the waveform based on the dominant frequency
    line.set_color(color)

    # Update the text element every 3 seconds
    if counter[0] % 50 == 0:  # 50 frames * 20ms interval = 1 seconds
        text.set_text(f'Dominant Frequency: {dominant_freq:.2f} Hz')

    line.set_ydata(audio_data_mono)
    counter[0] += 1
    return line, text

# Start recording and display real-time waveform
print("Press 'q' to quit.")
with loopback:
    fig, ax = plt.subplots()
    x = np.linspace(0, 4096, 4096)  # Match blocksize
    line, = ax.plot(x, np.random.rand(4096))  # Match blocksize
    ax.set_ylim(-1, 1)

    # Remove all axes and markers
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust subplot parameters to remove margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Add a text element to display the dominant frequency
    text = ax.text(0.02, 0.93, '', transform=ax.transAxes, ha='left', va='top', fontsize=16, color='black')

    # Add text to the upper left corner of the visualization
    ax.text(0.02, 0.98, visualization_title, transform=ax.transAxes, ha='left', va='top', fontsize=16, color='black')

    # Set the window title
    plt.get_current_fig_manager().set_window_title(visualization_title)

    # Counter to control the update interval for the text
    counter = [0]

    ani = animation.FuncAnimation(fig, update_waveform, fargs=(line, loopback, text, counter), interval=20, cache_frame_data=False)

    plt.show()

    while True:
        if keyboard.is_pressed('q'):
            break
