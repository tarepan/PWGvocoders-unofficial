"""Inference runners"""

import os.path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.cuda
import yaml
import librosa
import librosa.feature

from .utils import load_model


Wave = NDArray[np.float32]
MelSpec = NDArray[np.float32]

class HiFiGAN:
    """HiFi-GAN runner"""
    def __init__(self, path_ckpt: str):
        # Checkpoint and Config
        path_conf = os.path.join(os.path.dirname(path_ckpt), "config.yml")
        with open(path_conf) as f:
            self._config = yaml.load(f, Loader=yaml.Loader)

        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load model parameters including normalization mean/var
        self.model = load_model(path_ckpt, self._config).eval().to(self._device)
        self.model.remove_weight_norm()

    def decode(self, mel_spec: MelSpec, exec_spec_norm: bool=True) -> Tuple[Wave, int]:
        """
        Convert a mel-spectrogram into a waveform.

        Args:
            mel_spec::(T, freq) - a normalized mel-spectrogram
            exec_spec_norm - Whether to implicitly normalize input spectrogram before conversion
        Returns:
            wave_npy - a waveform
            target_sr - sampling rate of the `wave_npy`
        """

        target_sr = self._config["sampling_rate"]

        with torch.no_grad():
            # (T, freq) -> (T', 1) -> (T',)
            wave = self.model.inference(mel_spec, normalize_before=exec_spec_norm).view(-1)
            wave_npy = wave.to('cpu').detach().numpy().copy()

        return wave_npy, target_sr

    def preprocess(self, wave: Wave, source_sr: int) -> Tuple[MelSpec, int]:
        """
        Preprocess a waveform into `self.decode` compatible mel-frequency log-amplitude spectrogram.

        Args:
            wave::[T,] - waveform
            source_sr - sampling rate of the `wave`
        Returns:
            logmel::[T, freq] - spectrogram
            target_sr - sampling rate of `decode` output waveform
        """
        target_sr = self._config["sampling_rate"]
        wave_resampled = librosa.resample(wave, source_sr, target_sr)
        mel = librosa.feature.melspectrogram(
            y=wave_resampled,
            sr=target_sr,
            n_fft=self._config["fft_size"],
            hop_length=self._config["hop_size"],
            win_length=self._config["win_length"],
            window=self._config["window"],
            pad_mode='reflect',
            power=1.0,
            fmin=0 if self._config["fmin"] is None else self._config["fmin"],
            fmax=target_sr / 2 if self._config["fmax"] is None else self._config["fmax"],
            n_mels=self._config["num_mels"],
        ).T
        # always eps == 1e-10
        mel_clip = np.maximum(1e-10, mel)
        # always log_base == 10.0
        logmel =  np.log10(mel_clip)

        return logmel, target_sr
