import soundcard as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

# Get the default microphone with loopback enabled
mic = sc.default_microphone()
loopback = mic.recorder(samplerate=48000, channels=2, blocksize=4096)  # Increase blocksize

# Set parameters
sample_rate = 48000  # Hz
channels = 2
min_freq = 20  # Minimum audible frequency in Hz
max_freq = 400  # Maximum audible frequency in Hz is 20,000 Hz. Narrow the range for better visualization.

visualization_title = 'Apres Ski Waveform Sampler Platter'

# Create a colormap that spans from violet to red
cmap = plt.get_cmap('turbo')
norm = Normalize(vmin=min_freq, vmax=max_freq)
scalar_map = ScalarMappable(norm=norm, cmap=cmap)

# Function to update the waveform in real-time
def update_waveform(frame, lines, loopback, text, counter):
    audio_data = loopback.record(numframes=4096)  # Match blocksize
    audio_data_mono = audio_data[:, 0] if channels > 1 else audio_data

    # Calculate the FFT to find the dominant frequency
    fft_data = np.fft.fft(audio_data_mono)
    freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft_data = np.abs(fft_data[:len(fft_data)//2])
    positive_phases = np.angle(fft_data[:len(fft_data)//2])

    dominant_freq = positive_freqs[np.argmax(positive_fft_data)]

    # Map the dominant frequency to a color in the colormap
    color = scalar_map.to_rgba(dominant_freq)

    # Update the waveform plot
    lines[0].set_ydata(audio_data_mono)
    lines[0].set_color(color)

    # Update the bar plot
    for bar, height in zip(lines[1], positive_fft_data):
        bar.set_height(height)
        bar.set_color(color)

    # Update the scatter plot
    lines[2].set_offsets(np.c_[positive_freqs, positive_fft_data])
    lines[2].set_color(color)

    # Update the polar plot
    lines[3].set_offsets(np.c_[positive_phases, positive_fft_data])
    lines[3].set_color(color)

    # Update the text element every 3 seconds
    if counter[0] % 150 == 0:  # 150 frames * 20ms interval = 3 seconds
        text.set_text(f'Dominant Frequency: {dominant_freq:.2f} Hz')

    counter[0] += 1
    return lines + [text]

# Start recording and display real-time waveform
print("Press 'q' to quit.")
with loopback:
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Waveform plot
    ax_waveform = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 4096, 4096)  # Match blocksize
    line_waveform, = ax_waveform.plot(x, np.random.rand(4096))  # Match blocksize
    ax_waveform.set_ylim(-1, 1)
    ax_waveform.axis('off')

    # Bar plot
    ax_bar = fig.add_subplot(gs[0, 1])
    bar_freqs = np.fft.fftfreq(4096, 1/sample_rate)[:4096 // 2]
    bar_heights = np.abs(np.fft.fft(np.random.rand(4096))[:4096 // 2])
    bar_plot = ax_bar.bar(bar_freqs, bar_heights, width=10)
    ax_bar.set_xlim(min_freq, max_freq)
    ax_bar.axis('off')

    # Scatter plot
    ax_scatter = fig.add_subplot(gs[1, 0])
    scatter_plot = ax_scatter.scatter(bar_freqs, bar_heights)
    ax_scatter.set_xlim(min_freq, max_freq)
    ax_scatter.axis('off')

    # Polar plot
    ax_polar = fig.add_subplot(gs[1, 1], projection='polar')
    polar_plot = ax_polar.scatter(np.angle(bar_freqs), bar_heights)
    ax_polar.set_ylim(0, max(bar_heights))

    # Adjust subplot parameters to remove margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.1, wspace=0.1)

    # Add a text element to display the dominant frequency
    text = ax_waveform.text(0.02, 0.93, '', transform=ax_waveform.transAxes, ha='left', va='top', fontsize=16, color='black')

    # Add text to the upper left corner of the visualization
    fig.suptitle(visualization_title, fontsize=16, color='black')

    # Set the window title
    plt.get_current_fig_manager().set_window_title(visualization_title)

    # Counter to control the update interval for the text
    counter = [0]

    ani = animation.FuncAnimation(fig, update_waveform, fargs=([line_waveform, bar_plot.patches, scatter_plot, polar_plot], loopback, text, counter), interval=20, cache_frame_data=False)

    plt.show()

    while True:
        if keyboard.is_pressed('q'):
            break