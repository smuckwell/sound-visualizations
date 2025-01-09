import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import keyboard
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Set
import pyaudio
from scipy.fft import fft, fftfreq

class AudioManager:
    """Handles audio input and processing"""
    def __init__(self):
        self.SAMPLE_RATE = 44100
        self.CHUNK = 1024
        self.FREQ_LIMIT_LOW = 20
        self.FREQ_LIMIT_HIGH = 8000
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=None
        )
        
        # Calculate frequency mask
        self.freqs = fftfreq(self.CHUNK, 1 / self.SAMPLE_RATE)[:self.CHUNK // 2]
        self.freq_mask = (self.freqs >= self.FREQ_LIMIT_LOW) & (self.freqs <= self.FREQ_LIMIT_HIGH)
        self.filtered_freqs = self.freqs[self.freq_mask]
        self.n_freqs = len(self.filtered_freqs)
        
    def cleanup(self):
        """Clean up audio resources"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio is not None:
            self.audio.terminate()

class Animation(ABC):
    """Abstract base class for animations"""
    def __init__(self, ax):
        self.ax = ax
        self.artists = []
        
    @abstractmethod
    def animate(self, frame: int) -> None:
        """Update the animation for the given frame"""
        pass
    
    @abstractmethod
    def get_title(self) -> str:
        """Return the title of the animation"""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources used by the animation"""
        if hasattr(self.ax, 'clear'):
            self.ax.clear()
        self.artists.clear()

class SineWaveAnimation(Animation):
    """Animated sine wave with moving phase"""
    def __init__(self, ax):
        super().__init__(ax)
        self.x = np.linspace(0, 10, 100)
        
    def animate(self, frame: int) -> None:
        self.ax.clear()
        y = np.sin(self.x + frame/10)
        line = self._plot_wave(y)
        self.artists = [line]
        self._set_plot_properties()
        
    def _plot_wave(self, y: np.ndarray):
        line, = self.ax.plot(self.x, y, 'b-', label='Moving Sine Wave')
        return line
        
    def _set_plot_properties(self) -> None:
        self.ax.set_title(self.get_title())
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.grid(True)
        self.ax.legend()
        
    def get_title(self) -> str:
        return 'Animation 1: Moving Sine Wave'

class ScatterAnimation(Animation):
    """Animated scatter plot with dynamic points"""
    def __init__(self, ax, n_points: int = 50):
        super().__init__(ax)
        self.n_points = n_points
        
    def animate(self, frame: int) -> None:
        self.ax.clear()
        x, y = self._calculate_positions(frame)
        colors = np.random.rand(self.n_points)
        scatter = self._plot_scatter(x, y, colors)
        self.artists = [scatter]
        self._set_plot_properties()
        
    def _calculate_positions(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.sin(frame/10) * np.random.rand(self.n_points) + np.cos(frame/20) * np.random.rand(self.n_points)
        y = np.cos(frame/10) * np.random.rand(self.n_points) + np.sin(frame/20) * np.random.rand(self.n_points)
        return x, y
        
    def _plot_scatter(self, x: np.ndarray, y: np.ndarray, colors: np.ndarray):
        return self.ax.scatter(x, y, c=colors, cmap='viridis', s=100, alpha=0.6)
        
    def _set_plot_properties(self) -> None:
        self.ax.set_title(self.get_title())
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.grid(True)
        
    def get_title(self) -> str:
        return 'Animation 2: Dancing Scatter Plot'

class AudioHexbinAnimation(Animation):
    """Animated hexbin plot of audio frequency data"""
    def __init__(self, ax):
        super().__init__(ax)
        self.HISTORY_SIZE = 100
        self.audio_manager = AudioManager()
        self.last_fft = np.zeros(self.audio_manager.n_freqs)
        self.frequency_history = []
        self.amplitude_history = []
        self.colorbar = None
        self.fig = ax.figure  # Store figure reference
        
    def animate(self, frame: int) -> None:
        try:
            # Read and process audio data
            data = np.frombuffer(
                self.audio_manager.stream.read(self.audio_manager.CHUNK, exception_on_overflow=False),
                dtype=np.float32
            )
            
            # Compute FFT and apply frequency mask
            fft_data = np.abs(fft(data))[:self.audio_manager.CHUNK // 2]
            fft_data = fft_data[self.audio_manager.freq_mask]
            
            # Apply smoothing
            smoothing_factor = 0.7
            fft_data = smoothing_factor * self.last_fft + (1 - smoothing_factor) * fft_data
            self.last_fft = fft_data.copy()
            
            # Normalize
            fft_max = np.max(fft_data)
            if fft_max > 0:
                fft_data = fft_data / fft_max
            
            # Store data points
            for freq, amp in zip(self.audio_manager.filtered_freqs, fft_data):
                self.frequency_history.append(freq)
                self.amplitude_history.append(amp)
            
            # Keep only recent history
            max_points = self.HISTORY_SIZE * self.audio_manager.n_freqs
            if len(self.frequency_history) > max_points:
                self.frequency_history = self.frequency_history[-max_points:]
                self.amplitude_history = self.amplitude_history[-max_points:]
            
            self._update_plot()
            
            # Force figure update
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Error in audio animation: {e}")
            
    def _update_plot(self):
        self.ax.clear()
        
        # Create hexbin plot
        hb = self.ax.hexbin(
            self.frequency_history,
            self.amplitude_history,
            gridsize=50,
            cmap='viridis',
            bins='log',
            extent=[self.audio_manager.FREQ_LIMIT_LOW, 
                   self.audio_manager.FREQ_LIMIT_HIGH, 0, 1]
        )
        
        # Update plot properties
        self._set_plot_properties()
        
        # Handle colorbar
        if self.colorbar is None:
            self.colorbar = plt.colorbar(hb, ax=self.ax)
        self.colorbar.ax.tick_params(colors='white')

    def _set_plot_properties(self) -> None:
        self.ax.set_title(self.get_title(), color='white')
        self.ax.set_facecolor('black')
        self.ax.set_xlim(self.audio_manager.FREQ_LIMIT_LOW, 
                        self.audio_manager.FREQ_LIMIT_HIGH)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Frequency (Hz)', color='white')
        self.ax.set_ylabel('Amplitude', color='white')
        self.ax.tick_params(colors='white')
        
        # Customize spines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        
    def cleanup(self) -> None:
        """Cleanup resources including audio"""
        super().cleanup()
        if hasattr(self, 'audio_manager'):
            self.audio_manager.cleanup()
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        
    def get_title(self) -> str:
        return 'Animation 3: Audio Frequency Visualization'
    
class AnimationManager:
    """Manages the creation and switching of animations"""
    def __init__(self):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.current_animation: Optional[Animation] = None
        self.current_anim_obj: Optional[FuncAnimation] = None
        self.animations = {
            '1': lambda: SineWaveAnimation(self.ax),
            '2': lambda: ScatterAnimation(self.ax),
            '3': lambda: AudioHexbinAnimation(self.ax)
        }
        self.active_keyboard_hooks: Set[str] = set()
        
        # Set up figure close event handler
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
    def start_animation(self, key: str) -> None:
        if key in self.animations:
            self._cleanup_current_animation()
            self.current_animation = self.animations[key]()
            self._create_animation()
            
    def _cleanup_current_animation(self) -> None:
        try:
            if (self.current_anim_obj is not None and 
                hasattr(self.current_anim_obj, 'event_source') and 
                self.current_anim_obj.event_source is not None):
                self.current_anim_obj.event_source.stop()
            
            if hasattr(self.ax, 'clear'):
                self.ax.clear()
            
            if self.current_animation is not None:
                self.current_animation.cleanup()
            
            self.current_anim_obj = None
            self.current_animation = None
            
            if plt.fignum_exists(self.fig.number):
                self.fig.canvas.draw_idle()
                
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")
            
    def _create_animation(self) -> None:
        self.current_anim_obj = FuncAnimation(
            self.fig,
            self.current_animation.animate,
            frames=None,
            interval=50 if not isinstance(self.current_animation, AudioHexbinAnimation) else 10,
            blit=False,
            cache_frame_data=False
        )
        
    def switch_visualization(self, key) -> None:
        self.start_animation(key.name)
        
    def _on_close(self, event):
        """Handle figure close event"""
        self.cleanup()
        
    def setup_keyboard_hooks(self) -> None:
        """Set up keyboard event handlers"""
        for key in self.animations.keys():
            try:
                keyboard.on_press_key(key, self.switch_visualization)
                self.active_keyboard_hooks.add(key)
            except Exception as e:
                print(f"Warning: Failed to set up keyboard hook for key {key}: {str(e)}")
                
    def cleanup_keyboard_hooks(self) -> None:
        """Safely remove keyboard hooks"""
        for key in list(self.active_keyboard_hooks):
            try:
                keyboard.unhook_key(key)
                self.active_keyboard_hooks.remove(key)
            except Exception as e:
                print(f"Warning: Failed to unhook key {key}: {str(e)}")
        
    def run(self) -> None:
        self.start_animation('1')
        self.setup_keyboard_hooks()
        plt.show()
        
    def cleanup(self) -> None:
        """Cleanup all resources when closing"""
        self._cleanup_current_animation()
        self.cleanup_keyboard_hooks()
        
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

def main():
    print("Interactive Animation Controls:")
    print("Press '1' for Animation 1: Moving Sine Wave")
    print("Press '2' for Animation 2: Dancing Scatter Plot")
    print("Press '3' for Animation 3: Audio Visualization")
    print("Close the window to exit")
    
    manager = AnimationManager()
    try:
        manager.run()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        manager.cleanup()

if __name__ == "__main__":
    main()