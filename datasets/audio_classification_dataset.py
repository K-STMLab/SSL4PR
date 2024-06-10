import os
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

        # ** articulation disabled for now **
        # self.articulation_feature_extractor = Articulation()
        # self.art_mean, self.art_variance = data_config["articulation_params"]["mean"], data_config["articulation_params"]["std"]
        # self.get_mean_variance_articulation_features()
        # print(f"Articulation features mean: {self.art_mean}")
        # print(f"Articulation features variance: {self.art_variance}")

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
            # stft = torch.stft(
            #     audio,
            #     n_fft=self.data_config.stft_params.n_fft,
            #     hop_length=self.data_config.stft_params.hop_length,
            #     win_length=self.data_config.stft_params.win_length,
            #     window=torch.hamming_window(self.data_config.stft_params.win_length),
            #     center=False,
            #     return_complex=True,
            # )
            # magnitude = torch.abs(stft)
            # if self.data_config.stft_params.use_delta_and_delta_delta:
            #     # compute deltas
            #     magnitude_delta = compute_deltas(magnitude)
            #     magnitude_delta_delta = compute_deltas(magnitude_delta)
            #     # concatenate
            #     magnitude = torch.cat((magnitude, magnitude_delta, magnitude_delta_delta), dim=0)
            # # permute 1, 0
            # magnitude = magnitude.permute(1, 0)
            # use librosa for stft
            stft = librosa.stft(
                y=audio.numpy(),
                n_fft=self.data_config.stft_params.n_fft,
                hop_length=self.data_config.stft_params.hop_length,
                win_length=self.data_config.stft_params.win_length,
                window="hamming",
                center=False,
            )
            # abs
            magnitude = torch.from_numpy(stft)
            magnitude = torch.abs(magnitude)
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
            # mfcc = torchaudio.transforms.MFCC(
            #     sample_rate=self.data_config.sample_rate,
            #     n_mfcc=self.data_config.stft_params.n_fcc,
            #     melkwargs={
            #         "n_fft": self.data_config.stft_params.n_fft,
            #         "hop_length": self.data_config.stft_params.hop_length,
            #         "win_length": self.data_config.stft_params.win_length,
            #         "center": False,
            #         "window_fn": torch.hamming_window,
            #         "n_mels": self.data_config.stft_params.n_mels,
            #     },
            # )(audio)SlidingWindow
            # if self.data_config.stft_params.use_delta_and_delta_delta:
            #     # compute deltas
            #     mfcc_delta = compute_deltas(mfcc)
            #     mfcc_delta_delta = compute_deltas(mfcc_delta)
            #     # concatenate
            #     mfcc = torch.cat((mfcc, mfcc_delta, mfcc_delta_delta), dim=0)
            # # permute 1, 0
            # mfcc = mfcc.permute(1, 0)
            # return mfcc
            
            # use librosa for mfcc
            mfcc = librosa.feature.mfcc(
                y=audio.numpy(),
                sr=self.data_config.sample_rate,
                n_mfcc=self.data_config.stft_params.n_fcc,
                n_fft=self.data_config.stft_params.n_fft,
                hop_length=self.data_config.stft_params.hop_length,
                win_length=self.data_config.stft_params.win_length,
                window="hamming",
                center=False,
            )
            mfcc = torch.from_numpy(mfcc)
            if self.data_config.stft_params.use_delta_and_delta_delta:
                # compute deltas
                mfcc_delta = compute_deltas(mfcc)
                mfcc_delta_delta = compute_deltas(mfcc_delta)
                # concatenate
                mfcc = torch.cat((mfcc, mfcc_delta, mfcc_delta_delta), dim=0)
            # permute 1, 0
            mfcc = mfcc.permute(1, 0)
            # print(f"MFCC shape: {mfcc.shape}")
            return mfcc
        elif self.data_config.stft_params.type == "mel":
            # Mel spectrogram
            # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            #     sample_rate=self.data_config.sample_rate,
            #     n_fft=self.data_config.stft_params.n_fft,
            #     hop_length=self.data_config.stft_params.hop_length,
            #     win_length=self.data_config.stft_params.win_length,
            #     center=False,
            #     window_fn=torch.hamming_window,
            #     n_mels=self.data_config.stft_params.n_mels,
            # )(audio)
            # if self.data_config.stft_params.use_delta_and_delta_delta:
            #     # compute deltas
            #     mel_spectrogram_delta = compute_deltas(mel_spectrogram)
            #     mel_spectrogram_delta_delta = compute_deltas(mel_spectrogram_delta)
            #     # concatenate
            #     mel_spectrogram = torch.cat((mel_spectrogram, mel_spectrogram_delta, mel_spectrogram_delta_delta), dim=0)
            # mel_spectrogram = mel_spectrogram.permute(1, 0)
            # return mel_spectrogram
            # use librosa for mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio.numpy(),
                sr=self.data_config.sample_rate,
                n_fft=self.data_config.stft_params.n_fft,
                hop_length=self.data_config.stft_params.hop_length,
                win_length=self.data_config.stft_params.win_length,
                window="hamming",
                center=False,
                n_mels=self.data_config.stft_params.n_mels,
            )
            mel_spectrogram = torch.from_numpy(mel_spectrogram)
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
                nfilts=self.data_config.stft_params.n_mels,
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
                nfilts=self.data_config.stft_params.n_mels,
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
        elif self.data_config.stft_params.type == "cochleagram":
            g_spec = erb_spectrogram(
                sig=audio.numpy(),
                fs=self.data_config.sample_rate,
                nfilts=self.data_config.stft_params.n_mels,
                nfft=self.data_config.stft_params.n_fft,
                window=SlidingWindow(
                    win_len=self.data_config.stft_params.win_length / self.data_config.sample_rate,
                    win_hop=self.data_config.stft_params.hop_length / self.data_config.sample_rate,
                    win_type="hamming"
                ),
            )
            g_spec = g_spec[0] # 0 is the cochleagram, 1 is the fourier transform
            g_spec = torch.from_numpy(g_spec)
            if self.data_config.stft_params.use_delta_and_delta_delta:
                # compute deltas
                g_spec_delta = compute_deltas(g_spec)
                g_spec_delta_delta = compute_deltas(g_spec_delta)
                # concatenate
                g_spec = torch.cat((g_spec, g_spec_delta, g_spec_delta_delta), dim=1)
            # g_spec = g_spec.permute(1, 0)
            return g_spec
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
            
        # speaker_id = os.path.basename(audio_path)
        # speaker_id = speaker_id.split("_")[0]
        # speaker_id = speaker_id.split("-")[0]
        # speaker_id = speaker_id.replace("readtext.wav", "")
        # item["speaker_name"] = speaker_id

        # debug
        # for key in item:
        #     if isinstance(item[key], torch.Tensor):
        #         print(f"{key}: {item[key].shape}")

        return item