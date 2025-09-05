import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

def create_pro_waveform(file_path, output_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    # Create time axis
    t = np.linspace(0, len(y)/sr, len(y))
    
    # High-quality figure
    plt.figure(figsize=(10, 2), dpi=200)
    ax = plt.gca()
    
    # Gradient fill
    ax.fill_between(t, y, color='#1DB954', alpha=0.3)
    
    # Waveform line
    ax.plot(t, y, color='#1DB954', lw=0.8)
    
    # Remove borders
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# Generate thumbnails
os.makedirs("thumbnails", exist_ok=True)

for file in os.listdir("sample_database"):
    if file.endswith(".wav"):
        input_path = os.path.join("sample_database", file)
        output_path = os.path.join("thumbnails", f"{file}.png")
        create_pro_waveform(input_path, output_path)