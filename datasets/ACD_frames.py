import os, glob
import torch
import random
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor
from typing import List, Dict
from torchaudio.functional import compute_deltas

from spafe.features.bfcc import bfcc as bfcc_extractor
from spafe.features.gfcc import gfcc as gfcc_extractor
from spafe.utils.preprocessing import SlidingWindow

class AudioClassificationDataset(Dataset):
    def __init__(
        self,
        original_audio_paths: List[str],
        original_labels: List[int],
        feature_extractor_name_or_path: str,
        class_mapping: Dict[str, int],
        data_config: Dict[str, int],
        is_test: bool = False,
        extended_eval: bool = False,
    ):
        super(AudioClassificationDataset, self).__init__()
        self.original_audio_paths = original_audio_paths
        self.original_labels = original_labels
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name_or_path, do_normalize=False)
        self.class_mapping = class_mapping
        self.data_config = data_config
        self.is_test = is_test
        self.extended_eval = extended_eval
        self.audio_paths, self.labels, self.identifiers = self._load_audio_paths_and_labels()

    def _load_audio_paths_and_labels(self):
        self.audio_paths = []
        self.labels = []
        self.identifiers = []
        # for each audio_path
        for i, ap in enumerate(self.original_audio_paths):
            # get the base name
            ap_base = ap.split("/")[-1]
            # remove wav extension
            ap_base = ap_base.replace(".wav", "")
            # find all files
            root_path = os.path.dirname(ap)
            if self.extended_eval:
                path_mappings = {
                    "/mnt/disk2/mlaquatra/pc_gita_ext/" : "/mnt/disk2/mlaquatra/pc_gita_ext_250ms/",
                    "/mnt/disk2/mlaquatra/pc_gita_ext_SE/" : "/mnt/disk2/mlaquatra/pc_gita_ext_SE_250ms/",
                    "/mnt/disk2/mlaquatra/pc_gita_ext_drv/" : "/mnt/disk2/mlaquatra/pc_gita_ext_drv_250ms/",
                }
                if "SE" in root_path:
                    k = "/mnt/disk2/mlaquatra/pc_gita_ext_SE/"
                    v = "/mnt/disk2/mlaquatra/pc_gita_ext_SE_250ms/"
                    new_root_path = root_path.replace(k, v)
                elif "drv" in root_path:
                    k = "/mnt/disk2/mlaquatra/pc_gita_ext_drv/"
                    v = "/mnt/disk2/mlaquatra/pc_gita_ext_drv_250ms/"
                    new_root_path = root_path.replace(k, v)
                else:
                    k = "/mnt/disk2/mlaquatra/pc_gita_ext/"
                    v = "/mnt/disk2/mlaquatra/pc_gita_ext_250ms/"
                    new_root_path = root_path.replace(k, v)
                
                
                
            else:
                new_root_path = root_path.replace("/mnt/disk2/mfturco/Data/PC-GITA_downsampled_16000Hz/", "/mnt/disk2/mlaquatra/pc_gita_16khz_250ms/")
            # in new root path find all files that contains the base name + other suffixes
            files = glob.glob(new_root_path + "/" + ap_base + "*")
            self.audio_paths.extend(files)
            self.labels.extend([self.original_labels[i]] * len(files))
            self.identifiers.extend([ap_base] * len(files))
            
        print("Number of audio files: ", len(self.audio_paths))
        print("Number of labels: ", len(self.labels))
        print("Number of identifiers: ", len(self.identifiers))

        return self.audio_paths, self.labels, self.identifiers
        

    def __len__(self):
        return len(self.audio_paths)

    def _load_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        # Resample if needed
        if sr != self.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)
            audio = resampler(audio)

        # convert to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # remove empty dimensions
        audio = torch.squeeze(audio)
        return audio

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio = self._load_audio(audio_path)
        item = {}

        # feature extraction
        try:
            features = self.feature_extractor(
                audio, 
                sampling_rate=self.feature_extractor.sampling_rate,
                max_length=int(self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds),
                padding=self.data_config.padding,
                truncation=self.data_config.truncation,
                return_tensors="pt",
                return_attention_mask=True,
            )
        except Exception as e:
            print(f"Processing {audio_path}... - audio shape: {audio.shape}")
            print(e)
            raise e
            

        item["input_values"] = features.input_values[0]

        if self.data_config.magnitude:
            magnitude = self._extract_stft_features(item["input_values"])
            # if double convert to float
            if magnitude.dtype == torch.float64:
                magnitude = magnitude.float()
            item["magnitudes"] = magnitude

        # check if binary classification from class_mapping
        if len(self.class_mapping) == 2:
            item["labels"] = torch.tensor(self.class_mapping[self.labels[idx]], dtype=torch.float)
        else:
            item["labels"] = torch.tensor(self.class_mapping[self.labels[idx]], dtype=torch.long)
            
        # add the identifier
        item["identifier"] = self.identifiers[idx]

        return item