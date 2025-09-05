import librosa
import numpy as np
from scipy.signal import find_peaks

def analyze_audio(file_path):
    # Load audio with enhanced settings
    y, sr = librosa.load(file_path, sr=44100, mono=True, duration=15)
    
    # --- Key Detection ---
    y_harmonic = librosa.effects.harmonic(y, margin=8.0)
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, 
        sr=sr,
        n_chroma=12,
        n_octaves=7,
        tuning=librosa.pitch_tuning(y_harmonic),
        bins_per_octave=48,
        threshold=0.1
    )
    
    # Key determination
    chroma_mean = np.mean(chroma, axis=1)
    key_index = np.argmax(chroma_mean)
    key_mapping = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_note = key_mapping[key_index]
    
    # Mode detection
    major_profile = librosa.util.normalize([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = librosa.util.normalize([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    key_mode = "Minor" if np.dot(chroma_mean, minor_profile) > np.dot(chroma_mean, major_profile) else "Major"
    key = f"{key_note} {key_mode}" if np.max(chroma_mean) > 0.45 else "Unknown"

    # --- Tempo/Beat Tracking ---
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", tightness=150)
    tempo = int(np.median(tempo)) if isinstance(tempo, np.ndarray) else int(tempo)
    
    # --- Transient Detection ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = find_peaks(onset_env, distance=32, prominence=0.5)[0]
    transients = librosa.frames_to_time(peaks, sr=sr).tolist()

    # --- Harmonic/Percussive Separation ---
    y_harmonic = librosa.effects.harmonic(y, margin=8)
    y_percussive = librosa.effects.percussive(y, margin=8)

    return (
        key, 
        tempo, 
        chroma, 
        beats.tolist(),
        transients,
        y_harmonic,
        y_percussive
    )