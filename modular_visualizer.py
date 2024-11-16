import soundcard as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import math

class AudioVisualizer:
    def __init__(self):
        # Audio setup
        self.mic = sc.default_microphone()
        self.loopback = self.mic.recorder(samplerate=48000, channels=2, blocksize=4096)
        
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
        
        # Data storage
        self.intensity_array = np.zeros((self.time_window, 4096 // 2))
        self.azimuthal_angle = 150
        self.active_plots = []
        self.axes = []
        self.texts = []
        
        # Initialize plotting
        self.setup_plot()
        
    def setup_plot(self):
        self.fig = plt.figure(figsize=(15, 15))
        plt.get_current_fig_manager().set_window_title(self.visualization_title)
        self.counter = 0
        
    def get_audio_data(self):
        audio_data = self.loopback.record(numframes=4096)
        audio_data_mono = audio_data[:, 0] if self.channels > 1 else audio_data
        
        # Calculate FFT
        fft_data = np.fft.fft(audio_data_mono)
        freqs = np.fft.fftfreq(len(fft_data), 1/self.sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft_data = np.abs(fft_data[:len(fft_data)//2]) * self.height_scale
        positive_phases = np.angle(fft_data[:len(fft_data)//2])
        
        dominant_freq = positive_freqs[np.argmax(positive_fft_data)]
        color = self.scalar_map.to_rgba(dominant_freq)
        
        return {
            'audio_data_mono': audio_data_mono,
            'positive_freqs': positive_freqs,
            'positive_fft_data': positive_fft_data,
            'positive_phases': positive_phases,
            'dominant_freq': dominant_freq,
            'color': color
        }

    def plot_waveform(self, ax, data):
        ax.clear()
        ax.plot(np.linspace(0, 4096, 4096), data['audio_data_mono'], color=data['color'])
        ax.set_ylim(-1, 1)
        ax.set_title("Waveform")
        ax.axis('off')

    def plot_barchart(self, ax, data):
        ax.clear()
        ax.bar(data['positive_freqs'], data['positive_fft_data'],
               width=10, color=data['color'])
        ax.set_xlim(self.min_freq, self.max_freq)
        ax.set_title("Frequency Spectrum")
        ax.axis('off')

    def plot_scatter(self, ax, data):
        ax.clear()
        ax.scatter(data['positive_freqs'], data['positive_fft_data'],
                  color=data['color'])
        ax.set_xlim(self.min_freq, self.max_freq)
        ax.set_title("Frequency Scatter")
        ax.axis('off')

    def plot_polar(self, ax, data):
        ax.clear()
        marker_sizes = data['positive_fft_data'] * 20
        radial_positions = data['positive_fft_data'] * 20
        polar_colors = self.scalar_map.to_rgba(data['positive_freqs'])
        ax.scatter(data['positive_phases'], radial_positions,
                  s=marker_sizes, c=polar_colors)
        ax.set_title("Polar Visualization")
        ax.axis('off')

    def plot_surface(self, ax, data):
        ax.clear()
        self.intensity_array[:-1] = self.intensity_array[1:]
        self.intensity_array[-1] = data['positive_fft_data']
        
        X, Y = np.meshgrid(data['positive_freqs'], np.arange(self.time_window))
        ax.plot_surface(X, Y, self.intensity_array, cmap='turbo')
        ax.view_init(elev=30, azim=self.azimuthal_angle)
        self.azimuthal_angle = (self.azimuthal_angle + 1) % 360
        ax.set_title("3D Surface")
        ax.axis('off')

    def plot_hexbin(self, ax, data):
        ax.clear()
        x = np.tile(data['positive_freqs'], 10)
        y = np.repeat(data['positive_fft_data'], 10)
        ax.hexbin(x, y, gridsize=30, cmap='turbo')
        ax.set_xlim(self.min_freq, self.max_freq)
        ax.set_title("Hexbin Density")
        ax.axis('off')

    def reorganize_layout(self):
        n = len(self.active_plots)
        if n == 0:
            return
        
        # Calculate grid dimensions
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        # Clear the figure
        self.fig.clear()
        self.axes = []
        gs = GridSpec(rows, cols, figure=self.fig)
        
        # Create new subplots
        for i, plot_type in enumerate(self.active_plots):
            row = i // cols
            col = i % cols
            
            if plot_type == '4':  # Polar plot
                ax = self.fig.add_subplot(gs[row, col], projection='polar')
            elif plot_type == '5':  # Surface plot
                ax = self.fig.add_subplot(gs[row, col], projection='3d')
            else:
                ax = self.fig.add_subplot(gs[row, col])
            
            self.axes.append(ax)
        
        # Add title
        if self.active_plots:
            self.fig.suptitle(self.visualization_title, fontsize=16, y=0.98)
        
        # Adjust layout
        self.fig.tight_layout()

    def check_keyboard(self):
        for key in ['1', '2', '3', '4', '5', '6']:
            if keyboard.is_pressed(key):
                if key not in self.active_plots:
                    self.active_plots.append(key)
                    self.reorganize_layout()
                    plt.pause(0.2)  # Debounce
        
        if keyboard.is_pressed('backspace'):
            if self.active_plots:
                self.active_plots.pop()
                self.reorganize_layout()
                plt.pause(0.2)  # Debounce

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
        print("Press 1-6 to add visualizations:")
        print("1: Waveform")
        print("2: Bar Chart")
        print("3: Scatter Plot")
        print("4: Polar Plot")
        print("5: Surface Plot")
        print("6: Hexbin Plot")
        print("Backspace: Remove last visualization")
        print("Press 'q' to quit.")
        
        with self.loopback:
            ani = animation.FuncAnimation(
                self.fig, self.update, interval=20,
                cache_frame_data=False
            )
            plt.show()
            
            while True:
                if keyboard.is_pressed('q'):
                    break

if __name__ == "__main__":
    visualizer = AudioVisualizer()
    visualizer.run()