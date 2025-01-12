import sys
import requests
from io import BytesIO
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import queue
import threading
import keyboard  # optional
from scipy.fft import fft, fftfreq
import random
import string
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --------------------------------------------------
#  1) COMMON AUDIO MANAGER
# --------------------------------------------------
class AudioManager:
    """
    A cross-platform AudioManager using sounddevice.
    It continuously records audio from the chosen device
    and stores blocks in a thread-safe queue.
    """
    def __init__(self, device=None, channels=1, samplerate=44100, blocksize=256):
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
        # Put the newest block into the queue
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

        - Discard older blocks if multiple accumulate, so only the newest is used.
        """
        with self.lock:
            while self.audio_queue.qsize() > 1:
                self.audio_queue.get_nowait()

            if self.audio_queue.empty():
                return np.empty((0, self.channels), dtype=np.float32)

            block = self.audio_queue.get_nowait()
            if num_frames is None:
                return block
            else:
                if block.shape[0] > num_frames:
                    block = block[:num_frames, :]
                return block


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
    """
    def __init__(self, fig, audio_manager, image_url):
        super().__init__(fig, audio_manager)
        self.image_url = image_url

        self.ax = self.fig.add_subplot(111)
        self.ax.set_visible(False)
        self.ax.axis('off')

        r = requests.get(self.image_url)
        img_data = BytesIO(r.content)
        self.img = plt.imread(img_data)

        self.img_height, self.img_width = self.img.shape[:2]
        self.aspect_ratio = self.img_width / self.img_height

        self.cmap = plt.get_cmap("turbo")
        self.color_index = 0.0
        self.instruction_text = None

    def activate(self):
        super().activate()
        self.ax.clear()
        self.ax.axis('off')

        self.ax.imshow(
            self.img,
            extent=[0, self.aspect_ratio, 0, 1],
            aspect='equal',
            alpha=0.5
        )
        self.ax.set_xlim(0, self.aspect_ratio)
        self.ax.set_ylim(0, 1)

        top_y = 0.85
        self.ax.text(
            self.aspect_ratio / 2.0, top_y,
            "APRÈS-SKI PARTY VISUALIZER",
            color="white", ha="center", va="center",
            fontsize=24, fontweight='bold',
            zorder=10
        )

        self.instruction_text = self.ax.text(
            self.aspect_ratio / 2.0, top_y - 0.15,
            "Press 1 for Dancing Polar Visualizer\n"
            "Press 2 for 3D Wireframe Visualizer\n"
            "Press 3 for Propeller Arms Visualizer\n"
            "Press 4 for Frequency Bar Chart\n"
            "Press '0' to come back here\n"
            "Press 'q' or 'Esc' to quit\n\n"
            "Press a letter key (A..Z) on this screen to select an input device\n"
            "(See device list in terminal)",
            color="yellow", ha="center", va="center",
            fontsize=16,
            zorder=10
        )

    def update_frame(self, frame):
        if not self.active:
            return
        self.color_index += 0.01
        if self.color_index >= 1.0:
            self.color_index = 0.0
        color = self.cmap(self.color_index)
        if self.instruction_text:
            self.instruction_text.set_color(color)


# --------------------------------------------------
#  4) FIRST VISUALIZATION (POLAR DANCING FFT)
# --------------------------------------------------
class DancingPolarVisualizer(VisualizationBase):
    def __init__(self, fig, audio_manager):
        super().__init__(fig, audio_manager)
        self.sample_rate = audio_manager.samplerate
        self.min_freq = 20
        self.max_freq = 6000
        self.height_scale = 4
        self.polar_radial_distance_scale = 30.0
        self.polar_marker_size_scale = 2000.0
        self.background_color = 'white'
        self.current_pos = [0.5, 0.5]
        self.target_pos = [0.5, 0.5]

        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_visible(False)
        self.ax.axis('off')
        self.fig.patch.set_facecolor(self.background_color)

        self.polar_plot = self.ax.scatter(np.zeros(1024), np.zeros(1024))
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
        block_len = len(mono)

        fft_data = np.abs(np.fft.fft(mono))
        half_len = block_len // 2
        fft_data = fft_data[:half_len]
        freqs = np.fft.fftfreq(block_len, 1 / self.sample_rate)[:half_len]

        if fft_data.size == 0:
            return
        dominant_freq = freqs[np.argmax(fft_data)]

        speed = np.clip(dominant_freq / self.max_freq, 0.01, 0.1)
        self._update_plot_position(speed)

        size_factor = 1 - (dominant_freq / self.max_freq) * 0.5
        self.ax.set_position([
            self.current_pos[0] - size_factor / 2,
            self.current_pos[1] - size_factor / 2,
            size_factor,
            size_factor
        ])

        marker_sizes = fft_data * self.polar_marker_size_scale
        radial_positions = fft_data * self.polar_radial_distance_scale
        polar_colors = self.scalar_map.to_rgba(freqs)

        phases = np.angle(np.fft.fft(mono))[:half_len]

        self.polar_plot.set_offsets(np.c_[phases, radial_positions])
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
        self.FREQ_LIMIT_LOW = 20
        self.FREQ_LIMIT_HIGH = 16000
        self.HISTORY_SIZE = 100
        self.MAX_ROTATION_SPEED = 20.0
        self.MIN_ROTATION_SPEED = 2.0

        self.z_axis_scaling = 0.5
        self.current_rotation = 0.0

        self.n_freqs = 64
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
        block_len = len(mono)

        fft_data = np.abs(np.fft.fft(mono))
        half_len = block_len // 2
        fft_data = fft_data[:half_len]
        freqs = np.fft.fftfreq(block_len, 1 / self.samplerate)[:half_len]

        mask = (freqs >= self.FREQ_LIMIT_LOW) & (freqs <= self.FREQ_LIMIT_HIGH)
        masked_fft_data = fft_data[mask]
        masked_freqs = freqs[mask]

        if masked_fft_data.size == 0:
            return

        if masked_fft_data.size > self.n_freqs:
            masked_fft_data = masked_fft_data[: self.n_freqs]
        else:
            temp = np.zeros(self.n_freqs)
            temp[: masked_fft_data.size] = masked_fft_data
            masked_fft_data = temp

        dom_freq = self._get_dominant_frequency(masked_fft_data, masked_freqs)
        rotation_speed = self._frequency_to_rotation_speed(dom_freq)

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

        db_spectrum = 20 * np.log10(smoothed_fft + 1e-6)
        db_spectrum = np.clip(db_spectrum, -80, 0)
        db_spectrum = (db_spectrum + 80) / 80
        db_spectrum *= self.z_axis_scaling * 4

        self.z = np.roll(self.z, -1, axis=0)
        self.z[-1, :] = db_spectrum

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
        self.ax.set_zlim(0.9, 1.9)


# --------------------------------------------------
#  7) THIRD VISUALIZATION (PROPELLER ARMS)
# --------------------------------------------------
class PropellerArmsVisualizer(VisualizationBase):
    """
    A radial "propeller" style visualization with 12 arms.

    - The arms rotate around the center at a speed based on the dominant frequency.
    - Each arm's arc length depends on the volume (amplitude).
    - The color of each dot cycles outward with the turbo colormap at a rate based on freq.
    - The radius of each dot has a sine-wave "undulation".
    - The dot size increases with distance from the center.
    - The overall radial extent is increased by 50%.
    """

    def __init__(self, fig, audio_manager):
        super().__init__(fig, audio_manager)

        self.samplerate = audio_manager.samplerate
        self.num_arms = 12
        self.max_freq = 8000
        self.min_freq = 20

        self.background_color = 'white'
        # We'll track rotation, color, and sine wave phases.
        self.rotation_angle = 0.0
        self.color_phase = 0.0
        self.sine_phase = 0.0

        # Create a polar Axes
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_visible(False)
        self.ax.axis('off')
        self.fig.patch.set_facecolor(self.background_color)

        # Create a single scatter plot
        self.scatter_plot = self.ax.scatter([], [])
        # Increase the total radial limit to 1.5 (instead of 1.0) => 50% bigger
        self.ax.set_ylim(0, 1.5)

        self.cmap = plt.get_cmap('turbo')
        self.norm = Normalize(vmin=0, vmax=1)
        self.scalar_map = ScalarMappable(norm=self.norm, cmap=self.cmap)

    def _get_dominant_frequency(self, fft_data, freq_axis):
        if len(fft_data) == 0 or np.all(fft_data == 0):
            return 0
        return freq_axis[np.argmax(fft_data)]

    def update_frame(self, frame):
        if not self.active:
            return

        # Read up to 512 frames from the queue
        audio_data = self.audio_manager.read_frames(num_frames=512)
        if audio_data.shape[0] < 1:
            return

        mono = audio_data[:, 0]
        block_len = len(mono)

        # Standard FFT logic
        fft_data = np.abs(np.fft.fft(mono))
        half_len = block_len // 2
        fft_data = fft_data[:half_len]
        freqs = np.fft.fftfreq(block_len, 1 / self.samplerate)[:half_len]

        if fft_data.size == 0:
            return

        # 1) Dominant freq
        dom_freq = self._get_dominant_frequency(fft_data, freqs)

        # 2) rotation speed
        rotation_speed = np.clip(dom_freq / self.max_freq, 0.01, 0.2)
        self.rotation_angle += rotation_speed

        # 3) color cycle speed
        color_speed = np.clip(dom_freq / self.max_freq, 0.02, 0.2)
        self.color_phase += color_speed

        # 4) wave speed for the sine undulation
        wave_speed = np.clip(dom_freq / self.max_freq, 0.02, 0.3)
        self.sine_phase += wave_speed

        # 5) amplitude => sets the maximum arm radius
        amplitude = np.sum(fft_data) / fft_data.size
        arc_radius = (0.5 + np.clip(amplitude / 200.0, 0, 0.5))* 2.75

        # We'll define how many points per arm
        points_per_arm = 30
        total_points = self.num_arms * points_per_arm

        angles = np.zeros(total_points)
        radii = np.zeros(total_points)
        color_vals = np.zeros(total_points)
        sizes = np.zeros(total_points)

        # Sine wave parameters
        wave_ampl = 0.08
        wave_stride = 0.5

        index = 0
        for i in range(self.num_arms):
            base_angle = 2.0 * np.pi * i / self.num_arms
            for j in range(points_per_arm):
                frac = j / (points_per_arm - 1)  # 0..1
                angles[index] = base_angle + self.rotation_angle

                # base radius
                base_r = frac * arc_radius
                # add sine wave
                r_wave = wave_ampl * np.sin(self.sine_phase + j * wave_stride)
                final_r = base_r + r_wave
                radii[index] = final_r

                # color cycles outward with color_phase
                cval = (self.color_phase + frac) % 1.0
                color_vals[index] = cval

                # Dot size grows with radius from center:
                # e.g. base=20, scale=120 => bigger difference
                sizes[index] = 20.0 + 240.0 * max(math.pow(final_r, 4), 0.0)

                index += 1

        # Convert color_vals to RGBA
        colors = self.scalar_map.to_rgba(color_vals)

        # Update the scatter
        self.scatter_plot.set_offsets(np.c_[angles, radii])
        self.scatter_plot.set_color(colors)
        self.scatter_plot.set_sizes(sizes)


# --------------------------------------------------
#  8) NEW VISUALIZATION (FREQUENCY BAR CHART)
# --------------------------------------------------
class FrequencyBarChartVisualizer(VisualizationBase):
    """
    A simple bar chart showing frequency bins of the current audio signal.

    - The FFT is split into n_bars bins (e.g. 32 or 64).
    - Each bar's height corresponds to the amplitude in that bin.
    - The bar's color is determined by its height (turbo colormap).
    """
    def __init__(self, fig, audio_manager):
        super().__init__(fig, audio_manager)
        self.samplerate = audio_manager.samplerate
        self.min_freq = 20.0
        self.max_freq = 8000.0
        self.n_bars = 32  # number of bars (bins)

        # Create a normal 2D axes
        self.ax = self.fig.add_subplot(111)
        self.ax.set_visible(False)
        self.ax.set_facecolor('white')
        self.ax.set_title("Frequency Bar Chart", color="black", fontsize=16)
        self.ax.set_xlim(0, self.n_bars)
        self.ax.set_ylim(0, 1)  # We can dynamically adjust or keep it [0..1]

        # Pre-create the bars; initially all 0 height
        self.x_data = np.arange(self.n_bars)
        self.bar_container = self.ax.bar(self.x_data, np.zeros(self.n_bars), 
                                         color='blue', width=0.8)

        # Turbo colormap
        self.cmap = plt.get_cmap('turbo')
        self.norm = Normalize(vmin=0.0, vmax=1.0)  # for amplitude scaling
        self.scalar_map = ScalarMappable(norm=self.norm, cmap=self.cmap)

    def update_frame(self, frame):
        if not self.active:
            return

        audio_data = self.audio_manager.read_frames(num_frames=1024)
        if audio_data.shape[0] < 1:
            return
        mono = audio_data[:, 0]
        block_len = len(mono)

        # Compute the fft and freq axis
        fft_data = np.abs(np.fft.fft(mono))
        half_len = block_len // 2
        fft_data = fft_data[:half_len]
        freqs = np.fft.fftfreq(block_len, 1 / self.samplerate)[:half_len]

        # Focus on range [min_freq..max_freq]
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        freqs = freqs[mask]
        fft_data = fft_data[mask]

        if fft_data.size == 0:
            return

        # Build equal-width frequency bins across [min_freq..max_freq]
        bin_edges = np.linspace(self.min_freq, self.max_freq, self.n_bars + 1)
        bar_heights = np.zeros(self.n_bars, dtype=np.float32)

        for i in range(self.n_bars):
            low = bin_edges[i]
            high = bin_edges[i+1]
            bin_mask = (freqs >= low) & (freqs < high)
            if np.any(bin_mask):
                bar_heights[i] = np.mean(fft_data[bin_mask])

        # Normalize bar_heights to [0..1] for convenience
        max_val = np.max(bar_heights)
        if max_val > 0:
            bar_heights /= max_val

        # Update the bars
        for rect, h in zip(self.bar_container, bar_heights):
            # Height
            rect.set_height(h)
            # Color via Turbo colormap
            color_val = self.scalar_map.to_rgba(h)
            rect.set_color(color_val)

        # Optionally, adjust y-limits so bars fit nicely
        self.ax.set_ylim(0, 1.0)


# --------------------------------------------------
#  9) VISUALIZATION MANAGER
# --------------------------------------------------
class VisualizationManager:
    def __init__(self):
        self.fig = plt.figure(figsize=(8, 6))

        self.available_devices = []
        all_devices = sd.query_devices()
        for i, d in enumerate(all_devices):
            if d["max_input_channels"] > 0:
                self.available_devices.append((i, d["name"]))

        letters = string.ascii_uppercase
        print("Available Audio Input Devices:")
        for i, (dev_index, dev_name) in enumerate(self.available_devices):
            if i >= 26:
                break
            print(f"  {letters[i]} -> index={dev_index}, name='{dev_name}'")

        self.audio_manager = AudioManager(
            device=None,
            channels=1,
            samplerate=44100,
            blocksize=256
        )
        self.audio_manager.start()

        # Title screen
        title_url = "https://soundvisualizations.blob.core.windows.net/media/2025.01.11-Apres_Ski_Party_Title.png"
        self.title_screen = TitleScreen(self.fig, self.audio_manager, title_url)

        # Existing visualizations
        self.viz1 = DancingPolarVisualizer(self.fig, self.audio_manager)
        self.viz2 = WireframeFFTVisualizer(self.fig, self.audio_manager)
        self.viz3 = PropellerArmsVisualizer(self.fig, self.audio_manager)

        # NEW: Fourth visualization "Frequency Bar Chart"
        self.viz4 = FrequencyBarChartVisualizer(self.fig, self.audio_manager)

        self.visualizations = [self.viz1, self.viz2, self.viz3, self.viz4]

        self.active_screen = self.title_screen
        self.active_screen.activate()

        self.anim = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=10,  # faster refresh
            blit=False
        )

        self.cid_keypress = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.manager.set_window_title("Après-Ski Party Visualizer")

    def on_key_press(self, event):
        if event.key in ['q', 'escape']:
            self.cleanup_and_close()

        if self.active_screen == self.title_screen:
            if event.key is not None:
                letter = event.key.upper()
                if letter in string.ascii_uppercase:
                    index = ord(letter) - ord('A')
                    if 0 <= index < len(self.available_devices):
                        dev_index, dev_name = self.available_devices[index]
                        print(f"Switching audio device to {dev_name} (index={dev_index})")
                        self.audio_manager.stop()
                        self.audio_manager.device = dev_index
                        self.audio_manager.start()

            if event.key == '1':
                self.switch_to(self.viz1)
            elif event.key == '2':
                self.switch_to(self.viz2)
            elif event.key == '3':
                self.switch_to(self.viz3)
            elif event.key == '4':
                self.switch_to(self.viz4)
        else:
            if event.key == '1':
                self.switch_to(self.viz1)
            elif event.key == '2':
                self.switch_to(self.viz2)
            elif event.key == '3':
                self.switch_to(self.viz3)
            elif event.key == '4':
                self.switch_to(self.viz4)
            elif event.key == '0':
                self.switch_to(self.title_screen)

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
#  10) MAIN
# --------------------------------------------------
if __name__ == "__main__":
    manager = VisualizationManager()
    print("Press '1', '2', '3', or '4' to switch from the splash screen to a visualization.")
    print("Press a letter key (A..Z) on the splash screen to select an input device (see list above).")
    print("Press '0' to return to the splash screen from a visualization.")
    print("Press 'q' or 'Esc' to quit.")
    manager.show()
