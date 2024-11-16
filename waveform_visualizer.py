import soundcard as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard

# Get the default microphone with loopback enabled
mic = sc.default_microphone()
loopback = mic.recorder(samplerate=48000, channels=2, blocksize=4096)  # Increase blocksize

# Set parameters
sample_rate = 48000  # Hz
channels = 2

# Function to update the waveform in real-time
def update_waveform(frame, line, loopback):
    audio_data = loopback.record(numframes=4096)  # Match blocksize
    audio_data_mono = audio_data[:, 0] if channels > 1 else audio_data
    line.set_ydata(audio_data_mono)
    return line,

# Start recording and display real-time waveform
print("Press 'q' to quit.")
with loopback:
    fig, ax = plt.subplots()
    x = np.linspace(0, 4096, 4096)  # Match blocksize
    line, = ax.plot(x, np.random.rand(4096))  # Match blocksize
    ax.set_ylim(-1, 1)

    # Add frequency markers
    freq_bins = np.fft.fftfreq(4096, 1/sample_rate)
    positive_freq_bins = freq_bins[:4096 // 2]  # Only take the positive frequencies
    num_ticks = 10
    tick_positions = np.linspace(0, 4096 // 2, num_ticks, endpoint=False, dtype=int)
    tick_labels = [f'{int(freq)} Hz' for freq in positive_freq_bins[tick_positions]]

    #ax.set_xticks(tick_positions)
    #ax.set_xticklabels(tick_labels)

    ani = animation.FuncAnimation(fig, update_waveform, fargs=(line, loopback), interval=20, cache_frame_data=False)

    plt.show()

    while True:
        if keyboard.is_pressed('q'):
            break
