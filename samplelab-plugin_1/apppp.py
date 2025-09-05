import tkinter as tk
from tkinter import ttk, filedialog
import librosa
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import midiutil
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class SampleLabPro:
    def __init__(self, root):
        self.root = root
        self.root.title("SampleLab Pro")
        self.root.geometry("1200x800")
        
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
        self.tempo = "N/A"

        self.create_header()
        self.create_waveform_display()
        self.create_chord_visualization()
        self.create_control_panel()

    def create_header(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.upload_btn = ttk.Button(header_frame, text="UPLOAD SAMPLE", 
                                   command=self.load_sample)
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        self.key_label = ttk.Label(header_frame, text="Key: -", 
                                 font=('Helvetica', 14, 'bold'),
                                 foreground=self.colors['active'])
        self.key_label.pack(side=tk.LEFT, padx=20)
        
        self.tempo_label = ttk.Label(header_frame, text="Tempo: - BPM", 
                                    font=('Helvetica', 14, 'bold'),
                                    foreground=self.colors['active'])
        self.tempo_label.pack(side=tk.LEFT)

    def create_waveform_display(self):
        waveform_frame = ttk.Frame(self.root)
        waveform_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.fig = Figure(figsize=(10, 3), facecolor=self.colors['background'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['background'])
        self.ax.tick_params(colors=self.colors['text'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=waveform_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_chord_visualization(self):
        chord_frame = ttk.Frame(self.root)
        chord_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.chord_fig = Figure(figsize=(10, 4), facecolor=self.colors['background'])
        self.chord_ax = self.chord_fig.add_subplot(111)
        self.chord_ax.set_facecolor(self.colors['background'])
        self.chord_ax.tick_params(colors=self.colors['text'])
        
        self.chord_canvas = FigureCanvasTkAgg(self.chord_fig, master=chord_frame)
        self.chord_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_control_panel(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(control_frame, text="Export MIDI", command=self.export_midi).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Export WAV", command=self.export_wav).pack(side=tk.LEFT, padx=10)

    def load_sample(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            self.analyze_audio(file_path)
            self.update_visualizations()

    def analyze_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        self.audio_data = y
        
        # Enhanced chroma analysis with fixed parameters
        y_harmonic = librosa.effects.harmonic(y, margin=4)
        self.chroma = librosa.feature.chroma_cqt(
            y=y_harmonic,
            sr=sr,
            n_chroma=12,
            hop_length=2048,
            n_octaves=6
        )
        
        # Calculate exact 10-second frame count
        frames_per_second = sr // 2048
        target_frames = 10 * frames_per_second
        self.chroma = self.chroma[:, :target_frames]
        self.times = librosa.frames_to_time(np.arange(self.chroma.shape[1]), sr=sr, hop_length=2048)
        
        chroma_avg = np.mean(self.chroma, axis=1)
        self.key = self.chord_labels[np.argmax(chroma_avg)]
        
        try:
            self.tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
        except AttributeError:
            self.tempo = librosa.beat.tempo(y=y, sr=sr)[0]

    def update_visualizations(self):
        self.ax.clear()
        self.chord_ax.clear()
        
        # Waveform plot with reference scaling
        display_samples = min(len(self.audio_data), 10 * self.sr)
        time_axis = np.linspace(0, 10, display_samples)
        self.ax.plot(time_axis, self.audio_data[:display_samples], color=self.colors['active'])
        self.ax.set_ylim(-0.4, 0.2)  # Match reference image
        self.ax.set_xlim(0, 10)
        self.ax.set_xticks(np.arange(0, 11, 1))
        self.ax.grid(color=self.colors['grid'], alpha=0.3, linestyle=':')
        
        # Chroma visualization with full 10-second coverage
        bin_width = 0.8
        for i, pitch in enumerate(reversed(self.chord_labels)):
            for t in np.arange(0, 10, 0.5):  # Half-second resolution
                mask = (self.times >= t) & (self.times < t+0.5)
                if np.any(mask):
                    segment = self.chroma[i, mask]
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
        
        # Axis configuration matching reference
        self.chord_ax.set_yticks(np.arange(12))
        self.chord_ax.set_yticklabels(reversed(self.chord_labels))
        self.chord_ax.set_xticks(np.arange(0, 11, 1))
        self.chord_ax.set_xlim(0, 10)
        self.chord_ax.grid(color=self.colors['grid'], alpha=0.3)
        
        self.key_label.config(text=f"Key: {self.key}")
        self.tempo_label.config(text=f"Tempo: {int(self.tempo)} BPM")
        
        self.canvas.draw()
        self.chord_canvas.draw()

    def export_midi(self):
        if self.chroma is not None:
            midi = midiutil.MIDIFile(1)
            midi.addTempo(0, 0, int(self.tempo))
            
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
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SampleLabPro(root)
    root.mainloop()