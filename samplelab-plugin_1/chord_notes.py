import librosa
import numpy as np

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_chords_and_notes(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    times = librosa.times_like(chroma)

    # Averaged chroma to find dominant notes
    note_indices = np.argmax(chroma, axis=0)
    note_names = [NOTE_NAMES[i] for i in note_indices]

    # Collapse repeated notes to show only changes (to simulate chords)
    display_notes = []
    display_times = []
    for i in range(len(note_names)):
        if i == 0 or note_names[i] != note_names[i-1]:
            display_notes.append(note_names[i])
            display_times.append(times[i])

    return display_notes, display_times
