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
        
        # Professional dark theme colors
        self.colors = {
            'background': '#121212',
            'secondary': '#1E1E1E',
            'text': '#E0E0E0',
            'highlight': '#1DB954',
            'accent': '#6C5CE7',
            'grid': '#303030',
            'border': '#000000'
        }
        
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', 
                           background=self.colors['background'],
                           foreground=self.colors['text'],
                           font=('Roboto', 10))
        
        self.style.configure('TButton', 
                           background=self.colors['secondary'],
                           borderwidth=0,
                           relief='flat',
                           padding=8)
        
        self.style.map('TButton',
                      background=[('active', self.colors['highlight'])],
                      foreground=[('active', self.colors['background'])])
        
        self.style.configure('TCheckbutton',
                           background=self.colors['background'],
                           indicatorcolor=self.colors['highlight'])
        
        # Artist presets
        self.artist_presets = {
            'Avitat Style': {'chop_interval': 2, 'color': '#FF4D4D', 'linestyle': '-'},
            'Kanye West': {'chop_interval': 4, 'color': self.colors['highlight'], 'linestyle': '--'},
            'Pharrell': {'chop_interval': 16, 'color': '#FFC857', 'linestyle': ':'},
            'Electronic': {'chop_interval': 8, 'color': self.colors['accent'], 'linestyle': '-.'}
        }

        # Audio analysis variables
        self.chord_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                            'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.max_display_time = 10
        self.sr = 22050
        self.audio_data = None
        self.chroma = None
        self.times = []
        self.key = "N/A"
        self.tempo = 0.0
        self.beats = np.array([])
        self.chop_points = []
        self.selected_artist = tk.StringVar(value='Kanye West')
        self.show_chops_var = tk.BooleanVar(value=True)

        self.create_header()
        self.create_main_display()
        self.create_control_panel()

    def create_header(self):
        header_frame = ttk.Frame(self.root, padding=(20, 15))
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Upload button
        self.upload_btn = ttk.Button(header_frame, 
                                   text="UPLOAD SAMPLE",
                                   command=self.load_sample)
        self.upload_btn.pack(side=tk.LEFT, padx=15)
        
        # Combined info display
        self.info_label = ttk.Label(header_frame,
                                  text="Key: -  |  Tempo: - BPM",
                                  font=('Roboto', 14, 'bold'),
                                  foreground=self.colors['highlight'])
        self.info_label.pack(side=tk.LEFT, padx=30)

    def create_main_display(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Waveform plot with dark styling
        waveform_frame = ttk.Frame(main_frame, relief='flat', borderwidth=1)
        waveform_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(12, 2.8), facecolor=self.colors['background'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['background'])
        self.ax.tick_params(colors=self.colors['text'])
        self.ax.spines['bottom'].set_color(self.colors['text'])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.18, top=0.92)
        self.canvas = FigureCanvasTkAgg(self.fig, master=waveform_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Chord visualization
        chord_frame = ttk.Frame(main_frame, relief='flat', borderwidth=1)
        chord_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chord_fig = Figure(figsize=(12, 3.8), facecolor=self.colors['background'])
        self.chord_ax = self.chord_fig.add_subplot(111)
        self.chord_ax.set_facecolor(self.colors['background'])
        self.chord_ax.tick_params(colors=self.colors['text'])
        self.chord_ax.spines['bottom'].set_color(self.colors['text'])
        self.chord_ax.spines['top'].set_visible(False)
        self.chord_ax.spines['right'].set_visible(False)
        self.chord_ax.spines['left'].set_visible(False)
        self.chord_fig.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.95)
        self.chord_canvas = FigureCanvasTkAgg(self.chord_fig, master=chord_frame)
        self.chord_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_control_panel(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Export controls
        ttk.Button(control_frame, 
                  text="Export MIDI", 
                  command=self.export_midi).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, 
                  text="Export WAV", 
                  command=self.export_wav).pack(side=tk.LEFT, padx=10)
        
        # Artist style buttons
        for artist in self.artist_presets:
            btn = ttk.Button(control_frame,
                           text=artist,
                           command=lambda a=artist: [self.selected_artist.set(a), self.generate_chops()])
            btn.pack(side=tk.LEFT, padx=5)
            if artist == self.selected_artist.get():
                btn.config(style='Accent.TButton')
        
        # Chop controls
        ttk.Button(control_frame, 
                  text="Generate Chops", 
                  command=self.generate_chops).pack(side=tk.LEFT, padx=15)
        
        ttk.Checkbutton(control_frame, 
                       text="Show Chops", 
                       variable=self.show_chops_var,
                       command=self.update_visualizations).pack(side=tk.LEFT, padx=15)

    def load_sample(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            self.analyze_audio(file_path)
            self.update_visualizations()

    def analyze_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        self.audio_data = y
        
        try:
            self.tempo = float(librosa.feature.rhythm.tempo(y=y, sr=sr)[0])
        except AttributeError:
            self.tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
        
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        self.beats = librosa.frames_to_time(beat_frames, sr=sr)
        
        y_harmonic = librosa.effects.harmonic(y, margin=4)
        self.chroma = librosa.feature.chroma_cqt(
            y=y_harmonic,
            sr=sr,
            n_chroma=12,
            hop_length=2048,
            n_octaves=6
        )
        
        target_frames = int(10 * sr / 2048)
        self.chroma = self.chroma[:, :target_frames]
        self.times = librosa.frames_to_time(np.arange(self.chroma.shape[1]), sr=sr, hop_length=2048)
        
        chroma_avg = np.mean(self.chroma, axis=1)
        self.key = self.chord_labels[np.argmax(chroma_avg)]

    def update_visualizations(self):
        self.ax.clear()
        self.chord_ax.clear()
        
        # Update info display
        self.info_label.config(text=f"Key: {self.key}  |  Tempo: {int(round(self.tempo))} BPM")
        
        # Waveform plot
        display_samples = min(len(self.audio_data), 10 * self.sr)
        time_axis = np.linspace(0, 10, display_samples)
        self.ax.plot(time_axis, self.audio_data[:display_samples], 
                    color=self.colors['highlight'], linewidth=0.8)
        self.ax.set_ylim(-0.4, 0.2)
        self.ax.set_xlim(0, 10)
        self.ax.set_xticks(np.arange(0, 11, 1))
        self.ax.grid(color=self.colors['grid'], alpha=0.3, linestyle=':')
        
        # Artist-specific chop lines
        if self.show_chops_var.get() and self.chop_points:
            style = self.artist_presets[self.selected_artist.get()]
            for chop in self.chop_points:
                if chop <= 10:
                    self.ax.axvline(x=chop, color=style['color'], 
                                   linestyle=style['linestyle'], 
                                   linewidth=1.5, alpha=0.9)
        
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
                    color = self.colors['highlight'] if intensity > 0.6 else self.colors['secondary']
                    
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
        
        # Axis styling
        self.chord_ax.set_yticks(np.arange(12))
        self.chord_ax.set_yticklabels(reversed(self.chord_labels), 
                                     fontsize=9)
        self.chord_ax.set_xticks(np.arange(0, 11, 1))
        self.chord_ax.set_xlim(0, 10)
        self.chord_ax.set_ylim(-0.5, 11.5)
        self.chord_ax.grid(color=self.colors['grid'], alpha=0.3)
        
        self.canvas.draw()
        self.chord_canvas.draw()

    def generate_chops(self):
        if self.beats.size > 0:
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