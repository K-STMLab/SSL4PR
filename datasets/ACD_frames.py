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
        # replace root_path with 
        # original: /mnt/disk2/mfturco/Data/PC-GITA_downsampled_16000Hz/
        # new: /mnt/disk2/mlaquatra/pc_gita_16khz_250ms
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

    '''
    def get_mean_variance_articulation_features(self):
        # get mean and variance of articulation features
        print(f"Getting mean and variance of articulation features... for {len(self.audio_paths)} files")
        means = []
        variances = []
        random_paths = random.sample(self.audio_paths, 1000)
        for audio_path in tqdm(random_paths, desc="Getting articulation features"):
            articulation = self.articulation_feature_extractor.extract_features_file(
                audio_path,
                static=True,
                plots=False, 
                fmt='torch',
            )[0]
            if torch.isnan(articulation).any():
                print(f"NaN in articulation features for {audio_path}")
                continue
            means.append(torch.mean(articulation, dim=0))
            variances.append(torch.var(articulation, dim=0))

        art_mean = torch.mean(torch.stack(means), dim=0)
        art_variance = torch.mean(torch.stack(variances), dim=0)
        return art_mean, art_variance
    '''

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
    
    # def _apply_augmentations(self, audio):
    #     '''
    #     If enabled in config, this functions apply augmentations to the audio signal
    #     Types of augmentations:
    #     - pitch shift - to make the model invariant to speaker-specific characteristics
    #     '''

    def _extract_stft_features(self, audio):
        if self.data_config.stft_params.type == "standard":
            stft = torch.stft(
                audio,
                n_fft=self.data_config.stft_params.n_fft,
                hop_length=self.data_config.stft_params.hop_length,
                win_length=self.data_config.stft_params.win_length,
                window=torch.hamming_window(self.data_config.stft_params.win_length),
                center=False,
                return_complex=True,
            )
            magnitude = torch.abs(stft)
            if self.data_config.stft_params.use_delta_and_delta_delta:
                # compute deltas
                magnitude_delta = compute_deltas(magnitude)
                magnitude_delta_delta = compute_deltas(magnitude_delta)
                # concatenate
                magnitude = torch.cat((magnitude, magnitude_delta, magnitude_delta_delta), dim=0)
            # permute 1, 0
            magnitude = magnitude.permute(1, 0)
            return magnitude
        
        elif self.data_config.stft_params.type == "mfcc":
            # MFCC spectrogram
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=self.data_config.sample_rate,
                n_mfcc=self.data_config.stft_params.n_fcc,
                melkwargs={
                    "n_fft": self.data_config.stft_params.n_fft,
                    "hop_length": self.data_config.stft_params.hop_length,
                    "win_length": self.data_config.stft_params.win_length,
                    "center": False,
                    "window_fn": torch.hamming_window,
                    "n_mels": self.data_config.stft_params.n_mels,
                },
            )(audio)
            if self.data_config.stft_params.use_delta_and_delta_delta:
                # compute deltas
                mfcc_delta = compute_deltas(mfcc)
                mfcc_delta_delta = compute_deltas(mfcc_delta)
                # concatenate
                mfcc = torch.cat((mfcc, mfcc_delta, mfcc_delta_delta), dim=0)
            # permute 1, 0
            mfcc = mfcc.permute(1, 0)
            return mfcc
        elif self.data_config.stft_params.type == "mel":
            # Mel spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.data_config.sample_rate,
                n_fft=self.data_config.stft_params.n_fft,
                hop_length=self.data_config.stft_params.hop_length,
                win_length=self.data_config.stft_params.win_length,
                center=False,
                window_fn=torch.hamming_window,
                n_mels=self.data_config.stft_params.n_mels,
            )(audio)
            if self.data_config.stft_params.use_delta_and_delta_delta:
                # compute deltas
                mel_spectrogram_delta = compute_deltas(mel_spectrogram)
                mel_spectrogram_delta_delta = compute_deltas(mel_spectrogram_delta)
                # concatenate
                mel_spectrogram = torch.cat((mel_spectrogram, mel_spectrogram_delta, mel_spectrogram_delta_delta), dim=0)
            mel_spectrogram = mel_spectrogram.permute(1, 0)
            return mel_spectrogram
        elif self.data_config.stft_params.type == "bfcc":
            # spafe.features.bfcc.bfcc(sig, fs=16000, num_ceps=13, pre_emph=0, pre_emph_coeff=0.97, win_len=0.025, win_hop=0.01, win_type='hamming', nfilts=26, nfft=512, low_freq=None, high_freq=None, scale='constant', dct_type=2, use_energy=False, lifter=22, normalize=1)[source]
            bfcc = bfcc_extractor(
                sig=audio.numpy(),
                fs=self.data_config.sample_rate,
                num_ceps=self.data_config.stft_params.n_fcc,
                window=SlidingWindow(
                    win_len=self.data_config.stft_params.win_length / self.data_config.sample_rate,
                    win_hop=self.data_config.stft_params.hop_length / self.data_config.sample_rate,
                    win_type="hamming"
                ),
                nfft=self.data_config.stft_params.n_fft,
            )
            bfcc = torch.from_numpy(bfcc)
            if self.data_config.stft_params.use_delta_and_delta_delta:
                # compute deltas
                bfcc_delta = compute_deltas(bfcc)
                bfcc_delta_delta = compute_deltas(bfcc_delta)
                # concatenate
                bfcc = torch.cat((bfcc, bfcc_delta, bfcc_delta_delta), dim=1)
            # bfcc = bfcc.permute(1, 0)
            return bfcc
        elif self.data_config.stft_params.type == "gfcc":
            gfcc = gfcc_extractor(
                sig=audio.numpy(),
                fs=self.data_config.sample_rate,
                num_ceps=self.data_config.stft_params.n_fcc,
                window=SlidingWindow(
                    win_len=self.data_config.stft_params.win_length / self.data_config.sample_rate,
                    win_hop=self.data_config.stft_params.hop_length / self.data_config.sample_rate,
                    win_type="hamming"
                ),
                nfft=self.data_config.stft_params.n_fft,
            )
            gfcc = torch.from_numpy(gfcc)
            if self.data_config.stft_params.use_delta_and_delta_delta:
                # compute deltas
                gfcc_delta = compute_deltas(gfcc)
                gfcc_delta_delta = compute_deltas(gfcc_delta)
                # concatenate
                gfcc = torch.cat((gfcc, gfcc_delta, gfcc_delta_delta), dim=1)
            # gfcc = gfcc.permute(1, 0)
            return gfcc
        else:
            raise ValueError("Unsupported spectrogram type: {}".format(self.data_config.stft_params.type))


    '''
    def _extract_articulation_features(self, audio_path):
        articulation_features = self.articulation_feature_extractor.extract_features_file(
            audio_path,
            static=True,
            plots=False, 
            fmt='torch',
        )

        # normalize features using mean and variance of training set
        articulation_features = (articulation_features - self.art_mean) / self.art_variance
        
        if torch.isnan(articulation_features).any():
            print(f"NaN in articulation features for {audio_path}")
            # torch has no attribute fillna
            articulation_features[torch.isnan(articulation_features)] = 0

        return articulation_features
    '''

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio = self._load_audio(audio_path)
        
        # random crop if longer than max length
        # if self.data_config.random_crop_longer_audio and not self.is_test:
        #     if audio.shape[0] > self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds:
        #         start = random.randint(0, audio.shape[0] - self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds)
        #         audio = audio[start:start + self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds]
                
        # if self.data_config.repeat_shorter_audio:
        #     # print("Repeating shorter audio")
        #     if audio.shape[0] < self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds:
        #         # repeat to march max length
        #         n_repeats = int(self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds / audio.shape[0]) + 1
        #         audio = audio.repeat(n_repeats)
        #         # trim to max length
        #         audio = audio[:self.feature_extractor.sampling_rate * self.data_config.max_length_in_seconds]

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

        # ** articulation disabled for now **
        # if self.data_config.articulation:
        #     articulation = self._extract_articulation_features(audio_path)
        #     item["articulation"] = articulation[0]
        #     item["articulation"] = item["articulation"].float()

        # check if binary classification from class_mapping
        if len(self.class_mapping) == 2:
            item["labels"] = torch.tensor(self.class_mapping[self.labels[idx]], dtype=torch.float)
        else:
            item["labels"] = torch.tensor(self.class_mapping[self.labels[idx]], dtype=torch.long)
            
        # add the identifier
        item["identifier"] = self.identifiers[idx]

        # debug
        # for key in item:
        #     if isinstance(item[key], torch.Tensor):
        #         print(f"{key}: {item[key].shape}")

        return item