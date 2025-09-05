import librosa
import numpy as np

def analyze_audio(file_path):
    # Load audio with enhanced settings
    y, sr = librosa.load(file_path, sr=44100, mono=True, duration=15)  # Focus on first 15 seconds
    
    # --- Key Detection (Precision Mode) ---
    # Isolate harmonic content aggressively
    y_harmonic = librosa.effects.harmonic(y, margin=10.0)  # Increased margin for better separation
    
    # Chroma features with advanced tuning correction
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, 
        sr=sr,
        n_chroma=12,
        n_octaves=7,
        tuning=librosa.pitch_tuning(y_harmonic),
        bins_per_octave=48  # Higher resolution
    )
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = librosa.util.normalize(chroma_mean)  # Normalize to 0-1
    
    # Confidence threshold check
    if np.max(chroma_mean) < 0.45:  # Reject weak key signals
        key = "Unknown"
    else:
        key_mapping = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_index = np.argmax(chroma_mean)
        key_note = key_mapping[key_index]
        
        # Key profile comparison with weighted scores
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        major_score = np.sum(chroma_mean * librosa.util.normalize(major_profile))
        minor_score = np.sum(chroma_mean * librosa.util.normalize(minor_profile))
        
        key_mode = "Minor" if minor_score > major_score else "Major"
        key = f"{key_note} {key_mode}"

    # --- Tempo Detection (Enhanced) ---
    # Dynamic programming beat tracking
    tempo, beats = librosa.beat.beat_track(
        y=y,
        sr=sr,
        units="time",
        tightness=150,  # Stricter beat alignment
        start_bpm=100   # Prior for hip-hop/trap
    )
    
    # Tempo post-processing
    if isinstance(tempo, np.ndarray):
        tempo = np.median(tempo)  # Use median for stability
    tempo = round(float(tempo), 2)
    
    # Common BPM rounding (90-180 range)
    if 90 <= tempo <= 180:
        tempo = round(tempo / 5) * 5  # Snap to nearest 5 BPM
    else:
        tempo = round(tempo)
    
    return key, tempo