import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import random
import keyboard
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Set
import pyaudio
from scipy.fft import fft, fftfreq


class PersistentSettings:
    """Manages persistent view and interaction settings"""
    def __init__(self):
        self.view_init = (30, -30)  # elevation, azimuth for 3D plots
        self.xlim = (-2, 2)
        self.ylim = (-2, 2)
        self.zlim = (0, 1)
        self.scale = 1.0
        self.aspect = 'auto'
        self.mouse_init = None
        self.button_pressed = None


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
        self.settings = PersistentSettings()
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Set up mouse and keyboard event handlers"""
        fig = self.ax.figure
        fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        
    def _on_mouse_press(self, event):
        if event.inaxes == self.ax:
            self.settings.button_pressed = event.button
            self.settings.mouse_init = (event.xdata, event.ydata)
    
    def _on_mouse_release(self, event):
        self.settings.button_pressed = None
        self.settings.mouse_init = None
    
    def _on_mouse_motion(self, event):
        if event.inaxes == self.ax and self.settings.button_pressed is not None:
            if hasattr(self.ax, 'view_init'):  # 3D axes
                if self.settings.button_pressed == 1:  # Left click - rotation
                    self.settings.view_init = (self.ax.elev, self.ax.azim)
            if self.settings.button_pressed == 3:  # Right click - pan
                if self.settings.mouse_init is not None and event.xdata is not None and event.ydata is not None:
                    dx = event.xdata - self.settings.mouse_init[0]
                    dy = event.ydata - self.settings.mouse_init[1]
                    
                    # Update axis limits directly
                    new_xlim = (self.settings.xlim[0] - dx, self.settings.xlim[1] - dx)
                    new_ylim = (self.settings.ylim[0] - dy, self.settings.ylim[1] - dy)
                    
                    self.settings.xlim = new_xlim
                    self.settings.ylim = new_ylim
    
    def _on_scroll(self, event):
        if event.inaxes == self.ax:
            scale_factor = 1.1 if event.button == 'up' else 0.9
            self.settings.scale *= scale_factor
            # Update limits proportionally
            self._update_limits(scale_factor)
    
    def _update_limits(self, scale_factor):
        """Update axis limits based on scale factor"""
        for lim_name in ['xlim', 'ylim', 'zlim']:
            if hasattr(self.settings, lim_name):
                current_lim = getattr(self.settings, lim_name)
                center = sum(current_lim) / 2
                range_val = current_lim[1] - current_lim[0]
                new_lim = (center - range_val / 2 * scale_factor,
                          center + range_val / 2 * scale_factor)
                setattr(self.settings, lim_name, new_lim)
    
    def _apply_persistent_settings(self):
        """Apply persistent settings to the current axes"""
        if hasattr(self.ax, 'view_init'):  # 3D axes
            self.ax.view_init(*self.settings.view_init)
        
        # Apply limits directly
        self.ax.set_xlim(self.settings.xlim)
        self.ax.set_ylim(self.settings.ylim)
        
        if hasattr(self.ax, 'set_zlim'):
            self.ax.set_zlim(self.settings.zlim)
    
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
        y = np.sin(self.x + frame / 10)
        
        line = self._plot_wave(self.x, y)
        self.artists = [line]
        self._set_plot_properties()
        self._apply_persistent_settings()
        
    def _plot_wave(self, x: np.ndarray, y: np.ndarray):
        line, = self.ax.plot(x, y, 'b-', label='Moving Sine Wave')
        return line
        
    def _set_plot_properties(self) -> None:
        self.ax.set_title(self.get_title())
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
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
        self._apply_persistent_settings()
        
    def _calculate_positions(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.sin(frame / 10) * np.random.rand(self.n_points) + np.cos(frame / 20) * np.random.rand(self.n_points)
        y = np.cos(frame / 10) * np.random.rand(self.n_points) + np.sin(frame / 20) * np.random.rand(self.n_points)
        return x, y
        
    def _plot_scatter(self, x: np.ndarray, y: np.ndarray, colors: np.ndarray):
        return self.ax.scatter(x, y, c=colors, cmap='viridis', s=100, alpha=0.6)
        
    def _set_plot_properties(self) -> None:
        self.ax.set_title(self.get_title())
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
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
            
            # Store data points without offsets
            freqs = self.audio_manager.filtered_freqs
            amplitudes = fft_data
            self.frequency_history.extend(freqs)
            self.amplitude_history.extend(amplitudes)
            
            # Keep only recent history
            max_points = self.HISTORY_SIZE * self.audio_manager.n_freqs
            if len(self.frequency_history) > max_points:
                self.frequency_history = self.frequency_history[-max_points:]
                self.amplitude_history = self.amplitude_history[-max_points:]
            
            self._update_plot()
            self._apply_persistent_settings()
            
            # Force figure update
            self.ax.figure.canvas.draw()
            self.ax.figure.canvas.flush_events()
            
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
            bins='log'
        )
        
        # Update plot properties
        self._set_plot_properties()
        
        # Handle colorbar
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = plt.colorbar(hb, ax=self.ax)
        self.colorbar.ax.tick_params(colors='white')
        
    def _set_plot_properties(self) -> None:
        self.ax.set_title(self.get_title(), color='white')
        self.ax.set_facecolor('black')
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
        try:
            super().cleanup()
            if hasattr(self, 'audio_manager'):
                self.audio_manager.cleanup()
            if self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None

        except Exception as e:
            print(f"Error in cleanup: {e}")
        
    def get_title(self) -> str:
        return 'Animation 3: Audio Frequency Visualization'

class PolarAnimation(AudioHexbinAnimation):
    """Animated polar plot with audio-driven movement using shared audio tools"""
    def __init__(self, ax):
        super().__init__(ax)
        
        # Inherit from AudioHexbinAnimation to use shared audio manager
        self.audio_manager = AudioManager()

        # Create a colormap
        self.cmap = plt.get_cmap('turbo')
        self.norm = Normalize(vmin=self.audio_manager.FREQ_LIMIT_LOW, vmax=self.audio_manager.FREQ_LIMIT_HIGH)
        self.scalar_map = ScalarMappable(norm=self.norm, cmap=self.cmap)

        self.polar_radial_distance_scale = 20.0
        self.polar_marker_size_scale = 320.0

        # Variables to track plot position and movement
        self.current_pos = [0.5, 0.5]  # Start at center of screen
        self.target_pos = [0.5, 0.5]
        self.movement_speed = 0.1

    def update_plot_position(self, current_pos, target_pos, speed):
        """Update plot position towards a new target"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.01:  # If close to target, get a new target
            return current_pos, self.get_new_target_position()

        # Move towards target
        current_pos[0] += dx * speed
        current_pos[1] += dy * speed

        return current_pos, target_pos

    def get_new_target_position(self):
        """Generate new random position"""
        padding = 0.2
        return [random.uniform(padding, 1-padding), random.uniform(padding, 1-padding)]

    def animate(self, frame: int) -> None:
        """Update the polar plot with audio data"""
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

            # Normalize the frequency data
            fft_max = np.max(fft_data)
            if fft_max > 0:
                fft_data = fft_data / fft_max

            # Find dominant frequency
            dominant_freq = self.audio_manager.filtered_freqs[np.argmax(fft_data)]

            # Update movement speed based on dominant frequency
            speed = np.clip(dominant_freq / self.audio_manager.FREQ_LIMIT_HIGH, 0.01, 0.1)

            # Update plot position
            self.current_pos, self.target_pos = self.update_plot_position(self.current_pos, self.target_pos, speed)

            # Update plot size inversely with frequency
            size_factor = 1 - (dominant_freq / self.audio_manager.FREQ_LIMIT_HIGH) * 0.5
            self.ax.set_position([self.current_pos[0] - size_factor/2, 
                                  self.current_pos[1] - size_factor/2, 
                                  size_factor, 
                                  size_factor])

            # Update polar plot
            marker_sizes = fft_data * self.polar_marker_size_scale
            radial_positions = fft_data * self.polar_radial_distance_scale
            polar_colors = self.scalar_map.to_rgba(self.audio_manager.filtered_freqs)

            self.ax.clear()
            self.ax.scatter(np.angle(fft_data), radial_positions, s=marker_sizes, c=polar_colors, cmap=self.cmap)
            self._apply_persistent_settings()

        except Exception as e:
            print(f"Error in audio animation: {e}")

    def get_title(self) -> str:
        return 'Animation 4: Dancing Polar Visualizer'

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
            '3': lambda: AudioHexbinAnimation(self.ax),
            '4': lambda: PolarAnimation(self.ax)
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
        
    def switch_visualization(self, event) -> None:
        if hasattr(event, 'name'):
            key = event.name
            self.start_animation(key)
        
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
        self.start_animation('3')
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
    print("Press '4' for Animation 4: Dancing Polar Visualizer")
    print("Use mouse to interact:")
    print("- Right click and drag to pan")
    print("- Scroll to zoom")
    print("- Left click and drag to rotate (3D plots only)")
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