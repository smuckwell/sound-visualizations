import soundcard as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import math
from collections import deque
import time

class AudioVisualizer:
    def __init__(self):
        # Audio setup with smaller blocksize for more frequent updates
        self.mic = sc.default_microphone()
        self.loopback = self.mic.recorder(samplerate=48000, channels=2, blocksize=2048)
        
        # Parameters
        self.sample_rate = 48000
        self.channels = 2
        self.min_freq = 20
        self.max_freq = 400
        self.height_scale = 4
        self.time_window = 100
        self.visualization_title = 'Apres Ski Waveform Sampler Platter'
        
        # Setup color mapping
        self.cmap = plt.get_cmap('turbo')
        self.norm = Normalize(vmin=self.min_freq, vmax=self.max_freq)
        self.scalar_map = ScalarMappable(norm=self.norm, cmap=self.cmap)
        
        # Data storage with smoothing buffers
        self.intensity_array = np.zeros((self.time_window, 1024))
        self.azimuthal_angle = 150
        self.active_plots = []
        self.axes = []
        
        # Smoothing buffers
        buffer_size = 3
        self.fft_buffer = deque(maxlen=buffer_size)
        self.color_buffer = deque(maxlen=buffer_size)
        self.freq_buffer = deque(maxlen=buffer_size)
        
        # FPS tracking
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Initialize plotting
        self.setup_plot()
        
    def setup_plot(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 15))
        self.fig.patch.set_facecolor('#1C1C1C')
        
        # Enable interactive mode and hardware acceleration
        plt.ion()
        
        # Set up the figure for better performance
        self.fig.set_animated(True)
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = 1.0
        plt.rcParams['agg.path.chunksize'] = 10000
        
        plt.get_current_fig_manager().set_window_title(self.visualization_title)
        self.counter = 0

    def get_audio_data(self):
        audio_data = self.loopback.record(numframes=2048)
        audio_data_mono = audio_data[:, 0] if self.channels > 1 else audio_data
        
        # Apply Hanning window for smoother FFT
        window = np.hanning(len(audio_data_mono))
        windowed_data = audio_data_mono * window
        
        # Calculate FFT with zero padding for better frequency resolution
        fft_data = np.fft.fft(windowed_data, n=4096)
        freqs = np.fft.fftfreq(4096, 1/self.sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft_data = np.abs(fft_data[:len(fft_data)//2]) * self.height_scale
        positive_phases = np.angle(fft_data[:len(fft_data)//2])
        
        # Smooth the FFT data using the buffer
        self.fft_buffer.append(positive_fft_data)
        smoothed_fft_data = np.mean(list(self.fft_buffer), axis=0)
        
        # Calculate dominant frequency from smoothed data
        dominant_freq = positive_freqs[np.argmax(smoothed_fft_data)]
        color = self.scalar_map.to_rgba(dominant_freq)
        
        # Smooth the color transitions
        self.color_buffer.append(color)
        smoothed_color = np.mean(list(self.color_buffer), axis=0)
        
        # Update FPS counter
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        return {
            'audio_data_mono': audio_data_mono,
            'positive_freqs': positive_freqs,
            'positive_fft_data': smoothed_fft_data,
            'positive_phases': positive_phases,
            'dominant_freq': dominant_freq,
            'color': smoothed_color,
            'fps': self.fps
        }

    def plot_waveform(self, ax, data):
        ax.clear()
        samples = np.linspace(0, 2048, 2048)
        ax.plot(samples, data['audio_data_mono'], color=data['color'], linewidth=1.5)
        ax.set_ylim(-1, 1)
        ax.set_title(f"Waveform ({data['fps']} FPS)", color='white')
        ax.axis('off')

    def plot_barchart(self, ax, data):
        ax.clear()
        mask = data['positive_freqs'] <= self.max_freq
        ax.bar(data['positive_freqs'][mask], 
               data['positive_fft_data'][mask],
               width=2, color=data['color'])
        ax.set_xlim(self.min_freq, self.max_freq)
        ax.set_title("Frequency Spectrum", color='white')
        ax.axis('off')

    def plot_scatter(self, ax, data):
        ax.clear()
        mask = data['positive_freqs'] <= self.max_freq
        ax.scatter(data['positive_freqs'][mask], 
                  data['positive_fft_data'][mask],
                  color=data['color'], s=2)
        ax.set_xlim(self.min_freq, self.max_freq)
        ax.set_title("Frequency Scatter", color='white')
        ax.axis('off')

    def plot_polar(self, ax, data):
        ax.clear()
        mask = data['positive_freqs'] <= self.max_freq
        marker_sizes = data['positive_fft_data'][mask] * 20
        radial_positions = data['positive_fft_data'][mask] * 20
        polar_colors = self.scalar_map.to_rgba(data['positive_freqs'][mask])
        ax.scatter(data['positive_phases'][mask], radial_positions,
                  s=marker_sizes, c=polar_colors, alpha=0.6)
        ax.set_title("Polar Visualization", color='white')
        ax.axis('off')

    def plot_surface(self, ax, data):
        ax.clear()
        self.intensity_array[:-1] = self.intensity_array[1:]
        self.intensity_array[-1] = data['positive_fft_data'][:1024]
        
        X, Y = np.meshgrid(
            data['positive_freqs'][:1024],
            np.arange(self.time_window)
        )
        
        ax.plot_surface(X, Y, self.intensity_array, cmap='turbo')
        ax.view_init(elev=30, azim=self.azimuthal_angle)
        self.azimuthal_angle = (self.azimuthal_angle + 2) % 360
        ax.set_title("3D Surface", color='white')
        ax.axis('off')

    def plot_hexbin(self, ax, data):
        ax.clear()
        mask = data['positive_freqs'] <= self.max_freq
        x = np.tile(data['positive_freqs'][mask], 5)
        y = np.repeat(data['positive_fft_data'][mask], 5)
        ax.hexbin(x, y, gridsize=30, cmap='turbo')
        ax.set_xlim(self.min_freq, self.max_freq)
        ax.set_title("Hexbin Density", color='white')
        ax.axis('off')

    def reorganize_layout(self):
        n = len(self.active_plots)
        if n == 0:
            return
        
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        self.fig.clear()
        self.axes = []
        gs = GridSpec(rows, cols, figure=self.fig)
        
        for i, plot_type in enumerate(self.active_plots):
            row = i // cols
            col = i % cols
            
            if plot_type == '4':
                ax = self.fig.add_subplot(gs[row, col], projection='polar')
            elif plot_type == '5':
                ax = self.fig.add_subplot(gs[row, col], projection='3d')
            else:
                ax = self.fig.add_subplot(gs[row, col])
            
            ax.set_facecolor('#2C2C2C')
            self.axes.append(ax)
        
        if self.active_plots:
            self.fig.suptitle(self.visualization_title, fontsize=16, color='white', y=0.98)
        
        self.fig.tight_layout()
        plt.draw()

    def check_keyboard(self):
        for key in ['1', '2', '3', '4', '5', '6']:
            if keyboard.is_pressed(key):
                if key not in self.active_plots:
                    self.active_plots.append(key)
                    self.reorganize_layout()
                    plt.pause(0.1)
        
        if keyboard.is_pressed('backspace'):
            if self.active_plots:
                self.active_plots.pop()
                self.reorganize_layout()
                plt.pause(0.1)

    def update(self, frame):
        self.check_keyboard()
        if not self.active_plots:
            return []
        
        data = self.get_audio_data()
        
        plot_functions = {
            '1': self.plot_waveform,
            '2': self.plot_barchart,
            '3': self.plot_scatter,
            '4': self.plot_polar,
            '5': self.plot_surface,
            '6': self.plot_hexbin
        }
        
        for ax, plot_type in zip(self.axes, self.active_plots):
            plot_functions[plot_type](ax, data)
        
        self.counter += 1
        return self.axes

    def run(self):
        print("\nHigh-Performance Audio Visualizer")
        print("--------------------------------")
        print("Controls:")
        print("1-6: Add visualizations")
        print("1: Waveform")
        print("2: Bar Chart")
        print("3: Scatter Plot")
        print("4: Polar Plot")
        print("5: Surface Plot")
        print("6: Hexbin Plot")
        print("Backspace: Remove last visualization")
        print("Q: Quit")
        print("--------------------------------\n")
        
        with self.loopback:
            ani = animation.FuncAnimation(
                self.fig, self.update,
                interval=1,  # Minimal interval
                blit=True,   # Use blitting for better performance
                cache_frame_data=False
            )
            plt.show()
            
            while True:
                if keyboard.is_pressed('q'):
                    break

if __name__ == "__main__":
    visualizer = AudioVisualizer()
    visualizer.run()