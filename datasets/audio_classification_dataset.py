import torch
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

from torch_audiomentations import Compose, PitchShift, PolarityInversion

class AudioClassificationDataset(Dataset):
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[int],
        feature_extractor_name_or_path: str,
        class_mapping: Dict[str, int],
        data_config: Dict[str, int],
        is_test: bool = False,
    ):
        super(AudioClassificationDataset, self).__init__()
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name_or_path, do_normalize=False)
        self.class_mapping = class_mapping
        self.data_config = data_config
        self.is_test = is_test
        self.is_whisper = True if "whisper" in feature_extractor_name_or_path else False
        if self.is_whisper:
            print("Using whisper feature extractor")
            
        self.apply_augmentations = Compose([
            PitchShift(min_transpose_semitones=-4, max_transpose_semitones=4, p=0.5, sample_rate=self.feature_extractor.sampling_rate),
            PolarityInversion(p=0.5, sample_rate=self.feature_extractor.sampling_rate),
        ])

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
        if not self.is_test:
            audio = self.apply_data_augmentations(audio, self.data_config, self.feature_extractor.sampling_rate)
        
        # random crop if longer than max length
        if self.data_config.random_crop_longer_audio and not self.is_test:
            if audio.shape[0] > self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds:
                start = random.randint(0, audio.shape[0] - self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds)
                audio = audio[start:start + self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds]
                
        if self.data_config.repeat_shorter_audio:
            # print("Repeating shorter audio")
            if audio.shape[0] < self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds:
                # repeat to march max length
                n_repeats = int(self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds / audio.shape[0]) + 1
                audio = audio.repeat(n_repeats)
                # trim to max length
                audio = audio[:self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds]

        item = {}

        # feature extraction
        features = self.feature_extractor(
            audio, 
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds,
            padding=self.data_config.padding,
            truncation=self.data_config.truncation,
            return_tensors="pt",
            return_attention_mask=True,
        )

        if self.is_whisper:
            item["input_features"] = features.input_features[0]
        else:
            item["input_values"] = features.input_values[0]

        if self.data_config.magnitude:
            magnitude = self._extract_stft_features(item["input_values"])
            # if double convert to float
            if magnitude.dtype == torch.float64:
                magnitude = magnitude.float()
            item["magnitudes"] = magnitude

        # check if binary classification from class_mapping
        if len(self.class_mapping) == 2: dtype = torch.float # probability
        else: dtype = torch.long # categorical
        item["labels"] = torch.tensor(self.class_mapping[self.labels[idx]], dtype=dtype)
        
        return item