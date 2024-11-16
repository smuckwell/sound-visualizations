import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.fftpack
from matplotlib.animation import FuncAnimation
from collections import deque
import time

# Seed prompt
# I need help writing some software using Python.
# I would like to capture the audio picked up by the microphone of this device. I would like to display the spectrum captured as it changes over time so it probably needs to be saved to some kind of internal array. The displayed plot should be a surface, using the turbo color map. I would like the displayed frequencies to be limited, between 20 hz and 5000 hz. I would like the plot to continuously rotate smoothly about the Z axis, with the rotation speed changing corresponding to the current dominant frequency. I would like the animation of the surface changing and rotation to be very smooth, so use techniques that can improve the display characteristics.

class AudioSpectrumVisualizer:
    def __init__(self, history_size=50):
        # Audio parameters
        self.CHUNK = 2048  # Larger chunk size for better frequency resolution
        self.RATE = 44100
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        
        # Frequency range parameters
        self.freq_min = 20
        self.freq_max = 5000
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Calculate frequency bins
        self.freqs = scipy.fftpack.fftfreq(self.CHUNK) * self.RATE
        self.mask = (self.freqs >= self.freq_min) & (self.freqs <= self.freq_max)
        self.freqs = self.freqs[self.mask]
        
        # Initialize spectrum history
        self.history_size = history_size
        self.spectrum_history = deque(maxlen=history_size)
        # Fill history with zeros initially
        for _ in range(history_size):
            self.spectrum_history.append(np.zeros_like(self.freqs))
        
        self.rotation_angle = 0
        self.dominant_freq = 0
        
        # Initialize plot
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 0.5])
        
        # Create initial surface plot
        self.times = np.arange(history_size)
        self.X, self.Y = np.meshgrid(self.times, self.freqs)
        initial_data = np.zeros((len(self.freqs), history_size))
        self.surface = [self.ax.plot_surface(
            self.X, self.Y, initial_data,
            cmap='turbo',
            rcount=100,  # Increase resolution
            ccount=100
        )]
        
        # Set labels and title
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Frequency (Hz)')
        self.ax.set_zlabel('Amplitude')
        self.ax.set_title('Real-time Audio Spectrum')
        
    def update(self, frame):
        try:
            # Read audio data
            data = np.frombuffer(self.stream.read(self.CHUNK, exception_on_overflow=False), dtype=np.float32)
            
            # Compute FFT and get magnitude spectrum
            fft = scipy.fftpack.fft(data)
            spectrum = np.abs(fft)[self.mask]
            
            # Apply log scaling and normalization
            spectrum = np.log10(spectrum + 1)
            spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
            
            # Update spectrum history
            self.spectrum_history.append(spectrum)
            
            # Find dominant frequency
            max_idx = np.argmax(spectrum)
            self.dominant_freq = self.freqs[max_idx]
            
            # Update rotation based on dominant frequency
            rotation_speed = (self.dominant_freq - self.freq_min) / (self.freq_max - self.freq_min) * 5
            self.rotation_angle += rotation_speed
            
            # Remove old surface
            if self.surface[0]:
                self.surface[0].remove()
            
            # Create new surface with updated data
            spectrum_matrix = np.array(list(self.spectrum_history)).T
            self.surface[0] = self.ax.plot_surface(
                self.X, self.Y, spectrum_matrix,
                cmap='turbo',
                rcount=100,
                ccount=100
            )
            
            # Update view angle
            self.ax.view_init(elev=30, azim=self.rotation_angle)
            
            return self.surface
            
        except Exception as e:
            print(f"Error in update: {e}")
            return self.surface
    
    def animate(self):
        # Create animation with explicit save_count
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=20,  # 50 FPS for smooth animation
            blit=True,
            cache_frame_data=False,  # Disable frame caching
            save_count=1000  # Limit number of cached frames
        )
        
        # Display the plot
        plt.show()
    
    def cleanup(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            plt.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    visualizer = AudioSpectrumVisualizer()
    try:
        visualizer.animate()
    except Exception as e:
        print(f"Error during animation: {e}")
    finally:
        visualizer.cleanup()