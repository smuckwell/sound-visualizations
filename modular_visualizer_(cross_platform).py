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
from matplotlib.collections import LineCollection

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
            "Press 5 for Waveform Line Plot\n"
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
    """
    def __init__(self, fig, audio_manager):
        super().__init__(fig, audio_manager)

        self.samplerate = audio_manager.samplerate
        self.num_arms = 12
        self.max_freq = 8000
        self.min_freq = 20

        self.background_color = 'white'
        # We'll track rotation, color, etc.
        self.rotation_angle = 0.0
        self.color_phase = 0.0

        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_visible(False)
        self.ax.axis('off')
        self.fig.patch.set_facecolor(self.background_color)

        self.scatter_plot = self.ax.scatter([], [])
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

        audio_data = self.audio_manager.read_frames(num_frames=512)
        if audio_data.shape[0] < 1:
            return
        mono = audio_data[:, 0]
        block_len = len(mono)

        fft_data = np.abs(np.fft.fft(mono))
        half_len = block_len // 2
        fft_data = fft_data[:half_len]
        freqs = np.fft.fftfreq(block_len, 1 / self.samplerate)[:half_len]

        if fft_data.size == 0:
            return

        dom_freq = self._get_dominant_frequency(fft_data, freqs)
        rotation_speed = np.clip(dom_freq / self.max_freq, 0.01, 0.2)
        self.rotation_angle += rotation_speed

        color_speed = np.clip(dom_freq / self.max_freq, 0.02, 0.2)
        self.color_phase += color_speed

        amplitude = np.sum(fft_data) / fft_data.size
        arc_radius = 0.5 + np.clip(amplitude / 200.0, 0, 0.5)

        points_per_arm = 30
        total_points = self.num_arms * points_per_arm

        angles = np.zeros(total_points)
        radii = np.zeros(total_points)
        color_vals = np.zeros(total_points)

        idx = 0
        for i in range(self.num_arms):
            base_angle = 2.0 * np.pi * i / self.num_arms
            for j in range(points_per_arm):
                frac = j / (points_per_arm - 1)
                angles[idx] = base_angle + self.rotation_angle
                radii[idx] = frac * arc_radius

                cval = (self.color_phase + frac) % 1.0
                color_vals[idx] = cval
                idx += 1

        colors = self.scalar_map.to_rgba(color_vals)

        self.scatter_plot.set_offsets(np.c_[angles, radii])
        self.scatter_plot.set_color(colors)
        self.scatter_plot.set_sizes(np.full(total_points, 40.0))


# --------------------------------------------------
#  8) UPDATED VISUALIZATION (FREQUENCY BAR CHART)
#     *No axes/ticks
#     *Mirrored horizontally & vertically
#     *Lowest freq in center, highest freq outward
# --------------------------------------------------
class FrequencyBarChartVisualizer(VisualizationBase):
    """
    A mirrored bar chart with the lowest frequencies at the center.
    Frequencies fan out left & right. Each set of bars is also mirrored
    above and below the x-axis. No axes or ticks displayed.
    """
    def __init__(self, fig, audio_manager):
        super().__init__(fig, audio_manager)
        self.samplerate = audio_manager.samplerate
        self.min_freq = 20.0
        self.max_freq = 8000.0
        self.n_bars = 32  # total frequency bins

        # Create a 2D axis, then hide all spines/ticks
        self.ax = self.fig.add_subplot(111)
        self.ax.set_visible(False)
        self.ax.set_facecolor('white')
        self.ax.axis('off')  # remove axes, ticks, labels entirely

        # We'll set up an x-range from -n_bars//2 to +n_bars//2
        # so that 0 is the center.
        half_n = self.n_bars // 2
        self.x_positions = np.arange(-half_n, half_n)  # e.g. if n_bars=32, x in [-16..+15]

        # We'll store data in an internal array of bar heights
        self.bar_heights = np.zeros(self.n_bars, dtype=np.float32)

        # We'll create a container of patches manually
        # Each freq bin will yield 2 "bars": top and bottom (mirrored)
        # We'll do that on the left side and the right side is also mirrored
        # Actually, to keep it simpler, we can just do one patch per bin
        # for the top, and one for the bottom, for a total of 2 * n_bars patches.
        # But we want the lower freq in the center => let's reorder the bins:
        #   bin 0 (lowest freq) => x=0
        #   bin 1 => x=-1
        #   bin 2 => x=+1
        #   bin 3 => x=-2
        #   bin 4 => x=+2
        # etc...
        # We'll create an array that maps bin index => x position
        # so bin_order[i] = x position for the i-th bin in ascending freq
        # Then we can place bars accordingly.
        bin_indices = np.arange(self.n_bars)
        # We'll reorder them in the pattern center(0), left(-1), right(+1), left(-2), right(+2) ...
        # e.g. [0,1,2,3,4,5, ...] => [0,1,2,3, ...]
        # We'll build a pattern to get x offsets.
        # For 32 bins, bin 0 is center, bin 1 is x=-1, bin 2 is x=+1, bin 3 is x=-2, ...
        # We'll define a small helper:
        def center_out_positions(n):
            # returns an array of x-locations of length n, in the pattern described
            # Example: n=5 => [0, -1, +1, -2, +2]
            coords = []
            offset = 1
            sign = -1
            current = 0
            coords.append(0)
            for i in range(1, n):
                coords.append(sign * offset)
                # toggle sign
                sign *= -1
                # after sign flips to +, we increment offset
                if sign < 0:
                    offset += 1
            return np.array(coords, dtype=int)

        self.bin_x_map = center_out_positions(self.n_bars)  # length 32, from center outward

        # Create bar patches by hand
        self.bar_patches_top = []
        self.bar_patches_bottom = []

        # We'll use a simple approach: we'll create a rectangle at each bin x
        # with width=0.8, height=some fraction in [0..1], and mirror it above/below y=0.
        self.width = 0.8
        for _ in range(self.n_bars):
            # We won't actually create them as "BarContainer" but raw rectangles we can manage
            rect_top = plt.Rectangle((0,0), self.width, 0, color='blue')
            rect_bottom = plt.Rectangle((0,0), self.width, 0, color='blue')
            self.ax.add_patch(rect_top)
            self.ax.add_patch(rect_bottom)
            self.bar_patches_top.append(rect_top)
            self.bar_patches_bottom.append(rect_bottom)

        # We also want to scale the x-limits so all patches fit:
        # If half_n=16, then x in [-16..15], each bar has width=0.8 => let's go about -16.5..15.5
        self.ax.set_xlim(-half_n - 0.5, half_n - 0.5)
        # We'll also scale y-limits to -1..+1 for now (since we only show bars up to 1 in height)
        self.ax.set_ylim(-1.0, 1.0)

        # Turbo colormap for coloring by bar height
        self.cmap = plt.get_cmap('turbo')
        self.norm = Normalize(vmin=0.0, vmax=1.0)
        self.scalar_map = ScalarMappable(norm=self.norm, cmap=self.cmap)

    def update_frame(self, frame):
        if not self.active:
            return

        audio_data = self.audio_manager.read_frames(num_frames=1024)
        if audio_data.shape[0] < 1:
            return
        mono = audio_data[:, 0]
        block_len = len(mono)

        # Compute FFT magnitude and freq axis
        fft_data = np.abs(np.fft.fft(mono))
        half_len = block_len // 2
        fft_data = fft_data[:half_len]
        freqs = np.fft.fftfreq(block_len, 1 / self.samplerate)[:half_len]

        # Filter to [min_freq..max_freq]
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        freqs = freqs[mask]
        fft_data = fft_data[mask]

        if fft_data.size == 0:
            # no data in that range
            return

        # Split into n_bars bins (ascending freq)
        bin_edges = np.linspace(self.min_freq, self.max_freq, self.n_bars + 1)
        bar_heights = np.zeros(self.n_bars, dtype=np.float32)
        for i in range(self.n_bars):
            low, high = bin_edges[i], bin_edges[i+1]
            bin_mask = (freqs >= low) & (freqs < high)
            if np.any(bin_mask):
                bar_heights[i] = np.mean(fft_data[bin_mask])

        # Normalize to [0..1]
        max_val = np.max(bar_heights)
        if max_val > 0:
            bar_heights /= max_val

        # Now update each patch. The i-th bin goes at x=self.bin_x_map[i]
        # Top bar: y=0..height
        # Bottom bar: y=0..-height
        # Color by the same height
        for i in range(self.n_bars):
            h = bar_heights[i]
            x_center = self.bin_x_map[i] - self.width/2.0

            color_val = self.scalar_map.to_rgba(h)
            rect_top = self.bar_patches_top[i]
            rect_bottom = self.bar_patches_bottom[i]

            # top rectangle
            rect_top.set_x(x_center)
            rect_top.set_y(0.0)
            rect_top.set_width(self.width)
            rect_top.set_height(h)
            rect_top.set_color(color_val)

            # bottom rectangle
            rect_bottom.set_x(x_center)
            rect_bottom.set_y(-h)
            rect_bottom.set_width(self.width)
            rect_bottom.set_height(h)
            rect_bottom.set_color(color_val)


# --------------------------------------------------
#  9) NEW VISUALIZATION (TIME WAVEFORM LINE)
#     *Points colored by distance from x-axis
#     *Line thickness also scaled by |y|
# --------------------------------------------------
class WaveformLineVisualizer(VisualizationBase):
    """
    Plots the time-domain waveform as a line. 
    - The color of each point corresponds to |amplitude|.
    - The line thickness increases with |amplitude|.
    Uses a LineCollection for per-segment control.
    """
    def __init__(self, fig, audio_manager):
        super().__init__(fig, audio_manager)

        self.samplerate = audio_manager.samplerate
        self.ax = self.fig.add_subplot(111)
        self.ax.set_visible(False)
        self.ax.axis('off')
        self.fig.patch.set_facecolor('white')

        # We create an (initially empty) LineCollection
        self.line_collection = LineCollection([], cmap='turbo', norm=Normalize(0,1))
        self.ax.add_collection(self.line_collection)

        # We'll fix the x-limits to show maybe 1024 samples 
        # (or you can do something more dynamic).
        self.ax.set_xlim(0, 256)
        # y-limits might be [-1..+1], or [-something.. +something]
        self.ax.set_ylim(-0.20, 0.20)

    def update_frame(self, frame):
        if not self.active:
            return

        audio_data = self.audio_manager.read_frames(num_frames=1024)
        if audio_data.shape[0] < 1:
            return

        mono = audio_data[:, 0]
        n = len(mono)
        if n < 2:
            return

        # x coords
        x = np.arange(n)
        y = mono

        # Build segments (n-1) line segments connecting consecutive points
        segments = []
        colors = []
        widths = []

        # We'll find absolute max amplitude for normalization
        # so that color = |y| / max_amp
        max_amp = np.max(np.abs(y)) if np.any(y) else 1e-6

        for i in range(n-1):
            x1, y1 = x[i],   y[i]
            x2, y2 = x[i+1], y[i+1]
            segments.append([[x1, y1],[x2, y2]])

            # We color this segment by the average amplitude of its endpoints
            avg_amp = 0.5*(abs(y1) + abs(y2))
            c = avg_amp / (max_amp + 1e-12)
            colors.append(c)

            # The thickness also scales with c (e.g. base=0.5.. thickness=0.5+something)
            lw = 1.0 + 16.0*c
            widths.append(lw)

        segments = np.array(segments)

        # Update the line collection
        self.line_collection.set_segments(segments)
        # Per-segment color
        self.line_collection.set_array(np.array(colors))
        # Per-segment linewidth
        self.line_collection.set_linewidths(widths)

        # Optionally, we can re-scale y-limits if you want auto scaling:
        # self.ax.set_ylim(y.min() * 1.1, y.max() * 1.1)


# --------------------------------------------------
#  10) VISUALIZATION MANAGER
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

        # Fourth: Mirrored Frequency Bar Chart
        self.viz4 = FrequencyBarChartVisualizer(self.fig, self.audio_manager)

        # Fifth: Time Waveform Line Visualizer
        self.viz5 = WaveformLineVisualizer(self.fig, self.audio_manager)

        self.visualizations = [self.viz1, self.viz2, self.viz3, self.viz4, self.viz5]

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
            elif event.key == '5':
                self.switch_to(self.viz5)
        else:
            if event.key == '1':
                self.switch_to(self.viz1)
            elif event.key == '2':
                self.switch_to(self.viz2)
            elif event.key == '3':
                self.switch_to(self.viz3)
            elif event.key == '4':
                self.switch_to(self.viz4)
            elif event.key == '5':
                self.switch_to(self.viz5)
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
#  11) MAIN
# --------------------------------------------------
if __name__ == "__main__":
    manager = VisualizationManager()
    print("Press '1', '2', '3', or '4' to switch from the splash screen to a visualization.")
    print("Press '5' to show the new Waveform Line Visualizer.")
    print("Press a letter key (A..Z) on the splash screen to select an input device (see list above).")
    print("Press '0' to return to the splash screen from a visualization.")
    print("Press 'q' or 'Esc' to quit.")
    manager.show()
