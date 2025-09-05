# save as create_test_samples.py
import numpy as np
import soundfile as sf

# Generate a sine wave (440Hz)
sr = 22050
duration = 3  # seconds
t = np.linspace(0, duration, int(sr * duration))
y = 0.5 * np.sin(2 * np.pi * 440 * t)

# Save to sample_database/
sf.write("sample_database/sine_440hz.wav", y, sr)