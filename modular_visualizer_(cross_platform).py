import sys
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import queue
import threading
import keyboard  # for quick key detection (optional on Windows)
from scipy.fft import fft, fftfreq
import random

# --------------------------------------------------
#  1) COMMON AUDIO MANAGER
# --------------------------------------------------
class AudioManager:
    """
    A cross-platform AudioManager using sounddevice.
    It continuously records audio from the chosen device
    and stores blocks in a thread-safe queue.
    """
    def __init__(self, device=None, channels=1, samplerate=44100, blocksize=1024):
        self.device = device
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize

        self.audio_queue = queue.Queue()
        self.stream = None
        self.lock = threading.Lock()
        self.running = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio Manager Status: {status}", file=sys.stderr)
        block = indata.copy()
        self.audio_queue.put(block)

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self.lock:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    def read_frames(self, num_frames=None):
        """
        Returns a numpy array of shape (frames, channels).
        If no data is available, returns an empty array.
        """
        with self.lock:
            if self.audio_queue.empty():
                return np.empty((0, self.channels), dtype=np.float32)

            if num_frames is None:
                block = self.audio_queue.get_nowait()
                return block

            blocks = []
            accumulated = 0
            while accumulated < num_frames and not self.audio_queue.empty():
                block = self.audio_queue.get_nowait()
                blocks.append(block)
                accumulated += block.shape[0]

            if len(blocks) == 0:
                return np.empty((0, self.channels), dtype=np.float32)

            data = np.concatenate(blocks, axis=0)
            if data.shape[0] > num_frames:
                data = data[:num_frames, :]
            return data

# --------------------------------------------------
#  2) BASE VISUALIZATION CLASS
# --------------------------------------------------
class VisualizationBase:
    def __init__(self, fig, audio_manager):
        self.fig = fig
        self.audio_manager = audio_manager
        self.active = False
        self.ax = None

    def activate(self):
        self.active = True
        if self.ax:
            self.ax.set_visible(True)

    def deactivate(self):
        self.active = False
        if self.ax:
            self.ax.set_visible(False)

    def update_frame(self, frame):
        pass

# --------------------------------------------------
#  3) TITLE SCREEN CLASS
# --------------------------------------------------
class TitleScreen(VisualizationBase):
    """
    Displays a background image and usage instructions.
    Waits for the user to press 1 or 2 (or q/Esc to quit).
    This class does not use audio, so audio_manager is unused.
    """
    def __init__(self, fig, audio_manager, image_url):
        super().__init__(fig, audio_manager)
        self.image_url = image_url

        # Create a full-figure Axes
        self.ax = self.fig.add_subplot(111)
        self.ax.set_visible(False)  # Hidden until activated
        self.ax.axis('off')

        # Download the image
        r = requests.get(self.image_url)
        img_data = BytesIO(r.content)
        self.img = plt.imread(img_data)

        # We'll show it in update_frame once active

    def activate(self):
        super().activate()
        # Show the background image
        self.ax.imshow(self.img, aspect='auto', extent=[0, 1, 0, 1], zorder=-1)
        self.ax.set_position([0, 0, 1, 1])  # fill entire figure

        # Place text instructions
        # Adjust font sizes/positions as desired
        self.ax.text(
            0.5, 0.85,
            "APRÈS-SKI PARTY VISUALIZER",
            color="white", ha="center", va="center",
            fontsize=26, fontweight='bold',
            zorder=10
        )
        self.ax.text(
            0.5, 0.6,
            "Press 1 for the Dancing Polar Visualizer\n"
            "Press 2 for the 3D Wireframe Visualizer\n\n"
            "Press 'q' or 'Esc' to quit",
            color="yellow",
            ha="center", va="center",
            fontsize=16,
            zorder=10
        )

    def update_frame(self, frame):
        # Title screen is static, so no updates needed.
        pass

# --------------------------------------------------
#  4) FIRST VISUALIZATION (POLAR DANCING FFT)
# --------------------------------------------------
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class DancingPolarVisualizer(VisualizationBase):
    def __init__(self, fig, audio_manager):
        super().__init__(fig, audio_manager)
        self.sample_rate = audio_manager.samplerate
        self.min_freq = 20
        self.max_freq = 16000
        self.height_scale = 4
        self.polar_radial_distance_scale = 20.0
        self.polar_marker_size_scale = 320.0
        self.background_color = 'white'
        self.current_pos = [0.5, 0.5]
        self.target_pos = [0.5, 0.5]

        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_visible(False)
        self.ax.axis('off')
        self.fig.patch.set_facecolor(self.background_color)

        self.initial_phases = np.zeros(1024)
        self.initial_radial = np.zeros(1024)
        self.polar_plot = self.ax.scatter(self.initial_phases, self.initial_radial)
        self.ax.set_ylim(0, 100)

        self.cmap = plt.get_cmap('turbo')
        self.norm = Normalize(vmin=self.min_freq, vmax=self.max_freq)
        self.scalar_map = ScalarMappable(norm=self.norm, cmap=self.cmap)

    def _get_new_target_position(self):
        padding = 0.2
        return [
            random.uniform(padding, 1 - padding),
            random.uniform(padding, 1 - padding)
        ]

    def _update_plot_position(self, speed):
        dx = self.target_pos[0] - self.current_pos[0]
        dy = self.target_pos[1] - self.current_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 0.01:
            self.target_pos = self._get_new_target_position()
        else:
            self.current_pos[0] += dx * speed
            self.current_pos[1] += dy * speed

    def update_frame(self, frame):
        if not self.active:
            return

        audio_data = self.audio_manager.read_frames(num_frames=1024)
        if audio_data.shape[0] < 1:
            return

        mono = audio_data[:, 0]
        fft_data = np.fft.fft(mono)
        freqs = np.fft.fftfreq(len(fft_data), 1 / self.sample_rate)
        half = len(fft_data) // 2

        positive_freqs = freqs[:half]
        positive_fft_data = np.abs(fft_data[:half]) * self.height_scale
        positive_phases = np.angle(fft_data[:half])

        dominant_freq = positive_freqs[np.argmax(positive_fft_data)]
        speed = np.clip(dominant_freq / self.max_freq, 0.01, 0.1)

        self._update_plot_position(speed)

        size_factor = 1 - (dominant_freq / self.max_freq) * 0.5
        self.ax.set_position([
            self.current_pos[0] - size_factor / 2,
            self.current_pos[1] - size_factor / 2,
            size_factor,
            size_factor
        ])

        marker_sizes = positive_fft_data * self.polar_marker_size_scale
        radial_positions = positive_fft_data * self.polar_radial_distance_scale
        polar_colors = self.scalar_map.to_rgba(positive_freqs)

        self.polar_plot.set_offsets(np.c_[positive_phases, radial_positions])
        self.polar_plot.set_sizes(marker_sizes)
        self.polar_plot.set_color(polar_colors)

# --------------------------------------------------
#  5) SECOND VISUALIZATION (3D WIREFRAME FFT)
# --------------------------------------------------
class WireframeFFTVisualizer(VisualizationBase):
    def __init__(self, fig, audio_manager):
        super().__init__(fig, audio_manager)
        self.samplerate = audio_manager.samplerate
        self.CHUNK = 1024
        self.FREQ_LIMIT_LOW = 1
        self.FREQ_LIMIT_HIGH = 5000
        self.HISTORY_SIZE = 100
        self.MAX_ROTATION_SPEED = 10.0
        self.MIN_ROTATION_SPEED = 1.0

        self.z_axis_scaling = 0.5
        self.current_rotation = 0.0

        self.n_freqs = 512
        self.x = np.linspace(-6, 6, self.n_freqs)
        self.y = np.linspace(-3, 3, self.HISTORY_SIZE)
        self.x, self.y = np.meshgrid(self.x, self.y)
        self.z = np.zeros((self.HISTORY_SIZE, self.n_freqs))
        self.last_fft = np.zeros(self.n_freqs)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_visible(False)
        self.ax.set_facecolor('white')
        self.ax.axis('off')

    def _get_dominant_frequency(self, fft_data, freq_axis):
        if len(fft_data) == 0 or np.all(fft_data == 0):
            return 0
        return freq_axis[np.argmax(fft_data)]

    def _frequency_to_rotation_speed(self, freq):
        if freq == 0:
            return self.MIN_ROTATION_SPEED
        norm = (freq - self.FREQ_LIMIT_LOW) / (self.FREQ_LIMIT_HIGH - self.FREQ_LIMIT_LOW)
        norm = np.clip(norm, 0, 1)
        speed = self.MIN_ROTATION_SPEED + (self.MAX_ROTATION_SPEED - self.MIN_ROTATION_SPEED) * norm
        return speed

    def update_frame(self, frame):
        if not self.active:
            return

        audio_data = self.audio_manager.read_frames(num_frames=self.CHUNK)
        if audio_data.shape[0] < 1:
            return

        mono = audio_data[:, 0]
        fft_data = np.abs(fft(mono))[: self.CHUNK // 2]
        freqs = fftfreq(self.CHUNK, 1 / self.samplerate)[: self.CHUNK // 2]

        mask = (freqs >= self.FREQ_LIMIT_LOW) & (freqs <= self.FREQ_LIMIT_HIGH)
        masked_fft_data = fft_data[mask]
        masked_freqs = freqs[mask]

        # Pad or slice
        if len(masked_fft_data) > self.n_freqs:
            masked_fft_data = masked_fft_data[: self.n_freqs]
        else:
            temp = np.zeros(self.n_freqs)
            temp[: len(masked_fft_data)] = masked_fft_data
            masked_fft_data = temp

        dominant_freq = self._get_dominant_frequency(masked_fft_data, masked_freqs)
        rotation_speed = self._frequency_to_rotation_speed(dominant_freq)

        self.current_rotation += rotation_speed
        if self.current_rotation >= 360:
            self.current_rotation -= 360

        smoothing_factor = 0.7
        smoothed_fft = smoothing_factor * self.last_fft + (1 - smoothing_factor) * masked_fft_data
        self.last_fft = smoothed_fft

        fft_max = np.max(smoothed_fft)
        if fft_max > 0:
            smoothed_fft = smoothed_fft / fft_max

        smoothed_fft = np.log1p(smoothed_fft) * self.z_axis_scaling * 4

        self.z = np.roll(self.z, -1, axis=0)
        self.z[-1, :] = smoothed_fft

        self.ax.clear()
        self.ax.axis('off')

        z_min, z_max = np.min(self.z), np.max(self.z)
        if z_min == z_max:
            normalized = np.zeros_like(self.z)
        else:
            normalized = (self.z - z_min) / (z_max - z_min)
        colors = plt.cm.turbo(normalized.ravel())

        self.ax.plot_wireframe(
            self.x,
            self.y,
            self.z,
            rcount=self.HISTORY_SIZE,
            ccount=self.n_freqs,
            linewidth=2.0,
            colors=colors
        )
        self.ax.view_init(30, self.current_rotation)
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-3, 3)
        self.ax.set_zlim(0, 1)

# --------------------------------------------------
#  6) VISUALIZATION MANAGER
# --------------------------------------------------
class VisualizationManager:
    def __init__(self):
        self.fig = plt.figure(figsize=(8, 6))

        # Single AudioManager
        self.audio_manager = AudioManager(
            device=None,
            channels=1,
            samplerate=44100,
            blocksize=1024
        )
        self.audio_manager.start()

        # Create Title Screen
        title_url = "https://soundvisualizations.blob.core.windows.net/media/2025.01.11-Apres_Ski_Party_Title.png"
        self.title_screen = TitleScreen(self.fig, self.audio_manager, title_url)

        # Create two visualizations
        self.viz1 = DancingPolarVisualizer(self.fig, self.audio_manager)
        self.viz2 = WireframeFFTVisualizer(self.fig, self.audio_manager)

        # Keep them in a list for easier management if desired
        self.visualizations = [self.viz1, self.viz2]

        # We'll track the current "screen" or "visualization" in a single variable
        self.active_screen = self.title_screen
        self.active_screen.activate()  # Show the title screen first

        # Create a single animation
        self.anim = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=20,  # ms
            blit=False
        )

        # Connect keypress
        self.cid_keypress = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Window title
        self.fig.canvas.manager.set_window_title("Après-Ski Party Visualizer")

    def on_key_press(self, event):
        # 'q' or 'escape' => quit
        if event.key in ['q', 'escape']:
            self.cleanup_and_close()

        # If currently on title screen
        if self.active_screen == self.title_screen:
            if event.key == '1':
                self.switch_to(self.viz1)
            elif event.key == '2':
                self.switch_to(self.viz2)
            return
        else:
            # If we're already in a visualization, we can also allow
            # switching (like the old behavior) if desired:
            if event.key == '1':
                self.switch_to(self.viz1)
            elif event.key == '2':
                self.switch_to(self.viz2)

    def switch_to(self, screen):
        if self.active_screen == screen:
            return
        self.active_screen.deactivate()
        self.active_screen = screen
        self.active_screen.activate()

    def update(self, frame):
        if self.active_screen:
            self.active_screen.update_frame(frame)

    def cleanup_and_close(self):
        if self.active_screen:
            self.active_screen.deactivate()
        self.audio_manager.stop()
        plt.close(self.fig)
        sys.exit(0)

    def show(self):
        plt.tight_layout()
        plt.show()

# --------------------------------------------------
#  7) MAIN
# --------------------------------------------------
if __name__ == "__main__":
    manager = VisualizationManager()
    print("Press '1' or '2' to switch between visualizations (when on title screen).")
    print("Press 'q' or 'Esc' to quit.")
    manager.show()
