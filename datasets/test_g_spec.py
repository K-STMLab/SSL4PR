import torch
import numpy as np
from spafe.features.gfcc import erb_spectrogram
from spafe.utils.preprocessing import SlidingWindow

# (1) - Create a simple signal
sig = np.random.rand(16000)

# (2) - Compute the ERB spectrogram
g_spec = erb_spectrogram(
    sig, 
    fs=16000,
    nfilts=80,
    nfft=400,
    window=SlidingWindow(
        win_len=400/16000,
        win_hop=160/16000,
        win_type="hamming"
    ),
)

# (3) - Print the shape of the ERB spectrogram
print(g_spec)
# convert to numpy array
print(len(g_spec))
print(g_spec[0].shape) # (n_frames, n_filts)
print(g_spec[1].shape)