import torch
import numpy as np
import random
import torchaudio
import librosa
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor
from typing import List, Dict
from torchaudio.functional import compute_deltas

from spafe.features.bfcc import bfcc as bfcc_extractor
from spafe.features.gfcc import gfcc as gfcc_extractor
from spafe.features.gfcc import erb_spectrogram
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks as gt_fbanks 
from spafe.utils.preprocessing import SlidingWindow

class AudioClassificationDataset(Dataset):
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[int],
        feature_extractor_name_or_path: str,
        class_mapping: Dict[str, int],
        data_config: Dict[str, int],
        is_test: bool = False,
        feat_type="articulation" # prosody, phonation, articulation
    ):
        super(AudioClassificationDataset, self).__init__()
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name_or_path, do_normalize=False)
        self.class_mapping = class_mapping
        self.data_config = data_config
        self.is_test = is_test
        self.feat_type = feat_type
        self.is_whisper = True if "whisper" in feature_extractor_name_or_path else False
        if self.is_whisper:
            print("Using whisper feature extractor")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        audio_path = audio_path.replace(
            "/mnt/disk2/mfturco/Data/PC-GITA_downsampled_16000Hz/", 
            "/mnt/disk2/mlaquatra/pc_gita_16khz_" + self.feat_type + "/"
        )
        
        # if self.feat_type == "articulation":
        #     # add Articulation folder
        #     base_folder = "/".join(audio_path.split("/")[:-1])
        #     file_name = audio_path.split("/")[-1]
        #     audio_path = f"{base_folder}/Articulation/{file_name}"
        # elif self.feat_type == "phonation":
        #     # add Phonation folder
        #     base_folder = "/".join(audio_path.split("/")[:-1])
        #     file_name = audio_path.split("/")[-1]
        #     audio_path = f"{base_folder}/Phonation/{file_name}"
        # elif self.feat_type == "prosody":
        #     # add Prosody folder
        #     base_folder = "/".join(audio_path.split("/")[:-1])
        #     file_name = audio_path.split("/")[-1]
        #     audio_path = f"{base_folder}/Prosody/{file_name}"
        # else:
        #     raise ValueError(f"feat_type {self.feat_type} not supported")
        
        # remove .wav and put .npy
        audio_path = audio_path.replace(".wav", ".npy")
        
        label = torch.tensor(self.class_mapping[self.labels[idx]], dtype=torch.float)
        # load npy vector - it is a numpy array
        try:
            features = np.load(audio_path)
        except:
            audio_path = audio_path.replace("/HC/", "/hc/")
            audio_path = audio_path.replace("/PD/", "/pd/")
            features = np.load(audio_path)
            
        features = torch.tensor(features, dtype=torch.float)
        # remove dim 0 that is 1
        features = features.squeeze(0)
        
        return {
            "features": features,
            "labels": label,
        }