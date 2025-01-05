import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import random
import sys

# Set parameters
sample_rate = 48000  # Hz
channels = 2
blocksize = 4096
min_freq = 20  # Minimum audible frequency in Hz
max_freq = 8000  # Maximum frequency for visualization
height_scale = 4  # Scale factor for visualization
polar_radial_distance_scale = 20.0
polar_marker_size_scale = 80.0

visualization_title = 'Dancing Polar Visualizer'

# Create a colormap
cmap = plt.get_cmap('turbo')
norm = Normalize(vmin=min_freq, vmax=max_freq)
scalar_map = ScalarMappable(norm=norm, cmap=cmap)

# Variables to track plot position and movement
current_pos = [0.5, 0.5]  # Start at center of screen
target_pos = [0.5, 0.5]
movement_speed = 0.1

# Buffer for audio data
audio_buffer = np.zeros(blocksize)

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    global audio_buffer
    audio_buffer = indata[:, 0]

def get_new_target_position():
    # Generate new random position with padding from edges
    padding = 0.2
    return [random.uniform(padding, 1-padding), random.uniform(padding, 1-padding)]

def update_plot_position(current_pos, target_pos, speed):
    # Move current position towards target position
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance < 0.01:  # If we're close enough to target, get new target
        return current_pos, get_new_target_position()
    
    # Move towards target
    current_pos[0] += dx * speed
    current_pos[1] += dy * speed
    
    return current_pos, target_pos

def update_waveform(frame, polar_plot, ax_polar, fig):
    global current_pos, target_pos, audio_buffer
    
    audio_data_mono = audio_buffer

    # Calculate FFT
    fft_data = np.fft.fft(audio_data_mono)
    freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft_data = np.abs(fft_data[:len(fft_data)//2]) * height_scale
    positive_phases = np.angle(fft_data[:len(fft_data)//2])

    # Find dominant frequency
    dominant_freq = positive_freqs[np.argmax(positive_fft_data)]
    
    # Update movement speed based on dominant frequency
    speed = np.clip(dominant_freq / max_freq, 0.01, 0.1)
    
    # Update plot position
    current_pos, target_pos = update_plot_position(current_pos, target_pos, speed)
    
    # Update plot size inversely with frequency
    size_factor = 1 - (dominant_freq / max_freq) * 0.5  # Will range from 0.5 to 1
    ax_polar.set_position([current_pos[0] - size_factor/2, 
                          current_pos[1] - size_factor/2, 
                          size_factor, 
                          size_factor])

    # Update the polar plot
    marker_sizes = positive_fft_data * polar_marker_size_scale
    radial_positions = positive_fft_data * polar_radial_distance_scale
    polar_colors = scalar_map.to_rgba(positive_freqs)
    
    polar_plot.set_offsets(np.c_[positive_phases, radial_positions])
    polar_plot.set_sizes(marker_sizes)
    polar_plot.set_color(polar_colors)

    return polar_plot,

def main():
    try:
        # List available audio devices
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        
        # Create polar plot
        ax_polar = fig.add_subplot(111, projection='polar')
        initial_phases = np.zeros(2048)
        initial_radial = np.zeros(2048)
        polar_plot = ax_polar.scatter(initial_phases, initial_radial)
        ax_polar.set_ylim(0, 100)
        ax_polar.axis('off')
        
        # Set window properties
        fig.patch.set_facecolor('white')
        plt.get_current_fig_manager().set_window_title(visualization_title)
        
        # Start the audio input stream
        stream = sd.InputStream(
            channels=channels,
            samplerate=sample_rate,
            blocksize=blocksize,
            callback=audio_callback
        )
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, 
            update_waveform, 
            fargs=(polar_plot, ax_polar, fig),
            interval=20, 
            cache_frame_data=False
        )
        
        with stream:
            print("\nStreaming started. Close the plot window to stop.")
            plt.show()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()