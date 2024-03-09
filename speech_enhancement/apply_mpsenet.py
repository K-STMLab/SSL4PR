from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from mp_datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MPNet
import soundfile as sf

from tqdm import tqdm

import os, glob
import numpy as np

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--root_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = json.load(open(os.path.join(os.path.split(args.checkpoint_file)[0], 'config.json')))
    configs = AttrDict(configs)
    model = MPNet(configs).to(device)
    print("Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint_file, device)
    model.load_state_dict(checkpoint['generator'])
    model = model.eval()
    print("Model loaded successfully")
    
    files = glob.glob(args.root_folder + "**/*.wav", recursive=True)
    print("Found " + str(len(files)) + " files")

    with torch.no_grad():
        # MAINTAIN THE SAME FOLDER STRUCTURE
        for file in tqdm(files):
            # subfolder is the same but in the target folder
            subfolder = args.output_folder + file[len(args.root_folder):]
            # create target folder if it doesn't exist
            os.makedirs(os.path.dirname(subfolder), exist_ok=True)
            # load audio file
            # waveform, sample_rate = torchaudio.load(file)
            noisy_wav, _ = librosa.load(file, sr=configs.sampling_rate)
            # if longer than 10 seconds, skip
            if noisy_wav.shape[0] > 10 * configs.sampling_rate:
                original_shape = noisy_wav.shape
                clean_chunks = []
                final_chunk = False
                for start_idx in tqdm(range(0, noisy_wav.shape[0], 10 * configs.sampling_rate), desc="Chunking " + file):
                    end = start_idx + 10 * configs.sampling_rate if start_idx + 10 * configs.sampling_rate < noisy_wav.shape[0] else noisy_wav.shape[0]
                    
                    # check if the next chunk is the last and is too short (< 1000 ms)
                    next_start = end
                    if next_start < noisy_wav.shape[0] and noisy_wav.shape[0] - next_start < 16000:
                        print("Next chunk is too short, adding it to the current chunk")
                        end = noisy_wav.shape[0]
                        final_chunk = True
                    chunk = noisy_wav[start_idx:end]
                    noisy_chunk = torch.FloatTensor(chunk).to(device)
                    norm_factor = torch.sqrt(len(noisy_chunk) / torch.sum(noisy_chunk ** 2.0)).to(device)
                    noisy_chunk = (noisy_chunk * norm_factor).unsqueeze(0)
                    noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_chunk, configs.n_fft, configs.hop_size, configs.win_size, configs.compress_factor)
                    amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
                    audio_g = mag_pha_istft(amp_g, pha_g, configs.n_fft, configs.hop_size, configs.win_size, configs.compress_factor)
                    audio_g = audio_g / norm_factor
                    clean_chunks.append(audio_g.squeeze().cpu().numpy())
                    if final_chunk:
                        break
                clean_chunks = np.concatenate(clean_chunks)
                new_shape = clean_chunks.shape
                print(f"Original shape: {original_shape}, new shape: {new_shape}")
                subfolder = subfolder.replace(".wav", "")
                sf.write(subfolder + ".wav", clean_chunks, configs.sampling_rate, 'PCM_16')
            else:
                noisy_wav = torch.FloatTensor(noisy_wav).to(device)
                norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
                noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
                noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, configs.n_fft, configs.hop_size, configs.win_size, configs.compress_factor)
                amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
                audio_g = mag_pha_istft(amp_g, pha_g, configs.n_fft, configs.hop_size, configs.win_size, configs.compress_factor)
                audio_g = audio_g / norm_factor
                # save audio in wav format
                # remove .wav
                subfolder = subfolder.replace(".wav", "")
                sf.write(subfolder + ".wav", audio_g.squeeze().cpu().numpy(), configs.sampling_rate, 'PCM_16')
        
    print("Done")