import tkinter as tk
from tkinter import ttk, filedialog
import librosa
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import midiutil
import matplotlib.pyplot as plt
import warnings
import soundfile as sf

warnings.filterwarnings("ignore", category=FutureWarning)

class SampleLabPro:
    def __init__(self, root):
        self.root = root
        self.root.title("SampleLab Pro")
        self.root.geometry("1400x900")
        
        # Enhanced artist presets with specific styles
        self.artist_presets = {
            'J. Drilla': {'chop_interval': 1, 'color': '#FF0000', 'linestyle': '-'},
            'Kanye West': {'chop_interval': 4, 'color': '#1DB954', 'linestyle': '--'},
            'Pharrell': {'chop_interval': 16, 'color': '#FFD700', 'linestyle': ':'},
            'Electronic': {'chop_interval': 8, 'color': '#5757FF', 'linestyle': '-.'},
            'Custom': {'chop_interval': 2, 'color': '#FF5757', 'linestyle': '-'}
        }
        
        self.colors = {
            'background': '#1E1E1E',
            'text': '#FFFFFF',
            'grid': '#404040',
            'active': '#1DB954',
            'inactive': '#2A2A2A'
        }
        
        self.chord_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                            'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.max_display_time = 10
        self.sr = 22050
        self.audio_data = None
        self.chroma = None
        self.times = []
        self.key = "N/A"
        self.tempo = 0.0  # Store as float
        self.beats = np.array([])  # Initialize as numpy array
        self.chop_points = []
        self.selected_artist = tk.StringVar(value='Kanye West')
        self.show_chops_var = tk.BooleanVar(value=True)

        self.create_header()
        self.create_main_display()
        self.create_control_panel()

    def create_header(self):
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.upload_btn = ttk.Button(header_frame, text="UPLOAD SAMPLE", 
                                   command=self.load_sample)
        self.upload_btn.pack(side=tk.LEFT, padx=20)
        
        analysis_frame = ttk.Frame(header_frame)
        analysis_frame.pack(side=tk.LEFT, padx=40)
        
        self.key_label = ttk.Label(analysis_frame, text="Key: -", 
                                 font=('Helvetica', 14, 'bold'),
                                 foreground=self.colors['active'])
        self.key_label.pack(anchor=tk.W)
        
        self.tempo_label = ttk.Label(analysis_frame, text="Tempo: - BPM", 
                                    font=('Helvetica', 14, 'bold'),
                                    foreground=self.colors['active'])
        self.tempo_label.pack(anchor=tk.W)

    def create_main_display(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Waveform display
        waveform_frame = ttk.Frame(main_frame)
        waveform_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(12, 2.5), facecolor=self.colors['background'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['background'])
        self.ax.tick_params(colors=self.colors['text'])
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=waveform_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Chord visualization
        chord_frame = ttk.Frame(main_frame)
        chord_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chord_fig = Figure(figsize=(12, 3.5), facecolor=self.colors['background'])
        self.chord_ax = self.chord_fig.add_subplot(111)
        self.chord_ax.set_facecolor(self.colors['background'])
        self.chord_ax.tick_params(colors=self.colors['text'])
        self.chord_fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
        self.chord_canvas = FigureCanvasTkAgg(self.chord_fig, master=chord_frame)
        self.chord_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_control_panel(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X, padx=15, pady=10)
        
        ttk.Button(control_frame, text="Export MIDI", command=self.export_midi).pack(side=tk.LEFT, padx=20)
        ttk.Button(control_frame, text="Export WAV", command=self.export_wav).pack(side=tk.LEFT, padx=20)
        
        ttk.Label(control_frame, text="Artist Style:").pack(side=tk.LEFT, padx=5)
        artist_menu = ttk.Combobox(control_frame, textvariable=self.selected_artist,
                                  values=list(self.artist_presets.keys()))
        artist_menu.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Generate Chops", 
                  command=self.generate_chops).pack(side=tk.LEFT, padx=10)
        
        ttk.Checkbutton(control_frame, text="Show Chops", 
                       variable=self.show_chops_var,
                       command=self.update_visualizations).pack(side=tk.LEFT)

    def load_sample(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            self.analyze_audio(file_path)
            self.update_visualizations()

    def analyze_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        self.audio_data = y
        
        # Improved tempo detection with array handling
        try:
            self.tempo = float(librosa.feature.rhythm.tempo(y=y, sr=sr)[0])
        except AttributeError:
            self.tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
        
        # Get beat frames as numpy array
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        self.beats = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Chroma analysis
        y_harmonic = librosa.effects.harmonic(y, margin=4)
        self.chroma = librosa.feature.chroma_cqt(
            y=y_harmonic,
            sr=sr,
            n_chroma=12,
            hop_length=2048,
            n_octaves=6
        )
        
        # Limit to 10 seconds
        target_frames = int(10 * sr / 2048)
        self.chroma = self.chroma[:, :target_frames]
        self.times = librosa.frames_to_time(np.arange(self.chroma.shape[1]), sr=sr, hop_length=2048)
        
        # Key detection
        chroma_avg = np.mean(self.chroma, axis=1)
        self.key = self.chord_labels[np.argmax(chroma_avg)]

    def update_visualizations(self):
        self.ax.clear()
        self.chord_ax.clear()
        
        # Waveform plot
        display_samples = min(len(self.audio_data), 10 * self.sr)
        time_axis = np.linspace(0, 10, display_samples)
        self.ax.plot(time_axis, self.audio_data[:display_samples], color=self.colors['active'])
        self.ax.set_ylim(-0.4, 0.2)
        self.ax.set_xlim(0, 10)
        self.ax.set_xticks(np.arange(0, 11, 1))
        self.ax.grid(color=self.colors['grid'], alpha=0.3, linestyle=':')
        
        # Draw artist-specific chop lines
        if self.show_chops_var.get() and self.chop_points:
            style = self.artist_presets[self.selected_artist.get()]
            for chop in self.chop_points:
                if chop <= 10:
                    self.ax.axvline(x=chop, color=style['color'], 
                                   linestyle=style['linestyle'], alpha=0.8)
        
        # Chord visualization
        bin_width = 0.8
        for i in range(12):
            for t in np.arange(0, 10, 0.5):
                mask = (self.times >= t) & (self.times < t+0.5)
                if np.any(mask):
                    valid_indices = np.where(mask)[0]
                    if valid_indices[-1] >= self.chroma.shape[1]:
                        valid_indices = valid_indices[valid_indices < self.chroma.shape[1]]
                    segment = self.chroma[i, valid_indices]
                    intensity = np.mean(segment)
                    color = self.colors['active'] if intensity > 0.6 else self.colors['inactive']
                    
                    self.chord_ax.add_patch(
                        plt.Rectangle(
                            (t, i - bin_width/2),
                            width=0.5,
                            height=bin_width,
                            facecolor=color,
                            edgecolor=self.colors['background'],
                            linewidth=0.5
                        )
                    )
        
        # Axis configuration
        self.chord_ax.set_yticks(np.arange(12))
        self.chord_ax.set_yticklabels(reversed(self.chord_labels))
        self.chord_ax.set_xticks(np.arange(0, 11, 1))
        self.chord_ax.set_xlim(0, 10)
        self.chord_ax.set_ylim(-0.5, 11.5)
        self.chord_ax.grid(color=self.colors['grid'], alpha=0.3)
        
        # Update labels
        self.key_label.config(text=f"Key: {self.key}")
        self.tempo_label.config(text=f"Tempo: {int(round(self.tempo))} BPM")
        
        self.canvas.draw()
        self.chord_canvas.draw()

    def generate_chops(self):
        if self.beats.size > 0:  # Proper numpy array check
            interval = self.artist_presets[self.selected_artist.get()]['chop_interval']
            self.chop_points = self.beats[::interval].tolist()
            self.update_visualizations()
        else:
            print("No beats detected - cannot generate chops")

    def export_midi(self):
        if self.chroma is not None:
            midi = midiutil.MIDIFile(1)
            midi.addTempo(0, 0, int(round(self.tempo)))
            
            chroma_thresh = librosa.util.normalize(self.chroma, axis=0)
            for t_idx, t in enumerate(self.times):
                if t_idx >= chroma_thresh.shape[1]:
                    continue
                for note_idx in range(12):
                    if chroma_thresh[note_idx, t_idx] > 0.6:
                        midi.addNote(0, 0, 60 + note_idx, t, 0.5, 100)
            
            with open("chord_export.mid", "wb") as f:
                midi.writeFile(f)

    def export_wav(self):
        if self.audio_data is not None and len(self.chop_points) > 1:
            try:
                base_path = filedialog.asksaveasfilename(defaultextension=".wav")
                if base_path:
                    for i, (start, end) in enumerate(zip(self.chop_points[:-1], self.chop_points[1:])):
                        start_sample = int(start * self.sr)
                        end_sample = int(end * self.sr)
                        if 0 < start_sample < end_sample < len(self.audio_data):
                            chop = self.audio_data[start_sample:end_sample]
                            sf.write(f"{base_path}_chop_{i+1}.wav", chop, self.sr)
            except Exception as e:
                print(f"Export error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SampleLabPro(root)
    root.mainloop()