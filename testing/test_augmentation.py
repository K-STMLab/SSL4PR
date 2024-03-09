import os
import torch
import torchaudio
import numpy as np
from torch_audiomentations import Compose, PitchShift

def speaker_perturbation(waveform, sample_rate):
    # Create an instance of the transform we defined before
    augmenter = Compose(transforms=[
        PitchShift(min_transpose_semitones=-4, max_transpose_semitones=4, p=1.0, sample_rate=sample_rate)
    ])
    
    # Apply the transformation to the waveform
    waveform = waveform.unsqueeze(0)
    augmented_samples = augmenter(samples=waveform, sample_rate=sample_rate)
    augmented_samples = augmented_samples.squeeze(0)
    return augmented_samples, sample_rate


# Example usage:
# Load audio file using torchaudio
waveform, sample_rate = torchaudio.load('/mnt/disk2/mfturco/Data/PC-GITA_downsampled_16000Hz/sentences/laura/sin_normalizar/PD/AVPEPUDEA0035_laura.wav')

# save in samples/original.wav
os.makedirs('samples', exist_ok=True)
torchaudio.save('samples/original.wav', waveform, sample_rate)

# Apply speaker perturbation
perturbed_waveform, sample_rate = speaker_perturbation(waveform, sample_rate)
torchaudio.save('samples/perturbed.wav', perturbed_waveform, sample_rate)