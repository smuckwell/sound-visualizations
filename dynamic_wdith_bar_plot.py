import soundcard as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Get the default microphone with loopback enabled
mic = sc.default_microphone()
loopback = mic.recorder(samplerate=48000, channels=2, blocksize=4096)

# Set parameters
sample_rate = 48000  # Hz
channels = 2
min_freq = 20  # Minimum audible frequency in Hz
max_freq = 8000  # Maximum frequency to display
height_scale = 4  # Scale the height of the bars
max_bar_width = 20  # Maximum width of bars
min_bar_width = 2  # Minimum width of bars
min_color_scale_freq = 20  # Minimum frequency for color scaling
max_color_scale_freq = 400  # Maximum frequency for color scaling

visualization_title = 'Mirrored Audio Spectrum'

# Create a colormap that spans from violet to red
cmap = plt.get_cmap('turbo')
norm = Normalize(vmin=min_color_scale_freq, vmax=max_color_scale_freq)
scalar_map = ScalarMappable(norm=norm, cmap=cmap)

def update_visualization(frame, bars_top, bars_bottom, loopback):
    audio_data = loopback.record(numframes=4096)
    audio_data_mono = audio_data[:, 0] if channels > 1 else audio_data

    # Calculate the FFT to find the dominant frequency
    fft_data = np.fft.fft(audio_data_mono)
    freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft_data = np.abs(fft_data[:len(fft_data)//2]) * height_scale
    
    # Find dominant frequency index and value
    dominant_idx = np.argmax(positive_fft_data)
    dominant_freq = positive_freqs[dominant_idx]
    
    # Calculate base bar width (narrow)
    base_width = min_bar_width
    
    # Map the dominant frequency to a color
    color = scalar_map.to_rgba(dominant_freq)

    # Update both top and bottom bars
    for i, (bar_top, bar_bottom, height) in enumerate(zip(bars_top, bars_bottom, positive_fft_data)):
        # Set width based on whether this is the dominant frequency
        width = max_bar_width if i == dominant_idx else base_width
        
        # Update bar properties
        bar_top.set_height(height)
        bar_bottom.set_height(-height)  # Negative height for bottom bars
        bar_top.set_color(color)
        bar_bottom.set_color(color)
        bar_top.set_width(width)
        bar_bottom.set_width(width)
        
        # Center the bars based on their width
        current_x = bar_top.get_x()
        center_x = current_x + (max_bar_width / 2)  # Original center position
        new_x = center_x - (width / 2)  # New position to maintain center
        bar_top.set_x(new_x)
        bar_bottom.set_x(new_x)

    return bars_top + bars_bottom

# Start recording and display real-time visualization
print("Press 'q' to quit.")
with loopback:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the frequency bins
    bar_freqs = np.fft.fftfreq(4096, 1/sample_rate)[:4096 // 2]
    bar_heights = np.zeros(len(bar_freqs))
    
    # Create initial bars (both top and bottom)
    bars_top = ax.bar(bar_freqs, bar_heights, width=max_bar_width)
    bars_bottom = ax.bar(bar_freqs, -bar_heights, width=max_bar_width)  # Negative heights for bottom bars
    
    # Set up the plot
    ax.set_xlim(min_freq, max_freq)
    ax.set_ylim(-10, 10)  # Symmetric limits for top and bottom
    ax.axis('off')
    
    # Set up the window
    fig.suptitle(visualization_title, fontsize=16, color='black')
    plt.get_current_fig_manager().set_window_title(visualization_title)
    
    # Remove margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Create the animation
    ani = animation.FuncAnimation(
        fig, 
        update_visualization, 
        fargs=(bars_top, bars_bottom, loopback),
        interval=10, 
        cache_frame_data=False
    )
    
    plt.show()
    
    while True:
        if keyboard.is_pressed('q'):
            break

print("Done.")