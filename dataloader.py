# ============================================================
# dataloader.py
# """
# Overview

# This dataloader prepares audio data for training, validation, and testing
# a Voice Activity Detection (VAD) model.

# The pipeline performs the following:

# 1. **Load Speech and Noise Data**
#    - Speech samples are taken from the Speech Commands dataset.
#    - Background noise files are randomly selected from the noise directory.

# 2. **Create Synthetic Mixtures**
#    - Each training example combines clean speech with background noise
#      at a *random* Signal-to-Noise Ratio (SNR) between -20 dB and +20 dB.
#    - Short segments of silence are added before and after the speech.

# 3. **Compute Features**
#    - Converts the mixed waveform into a 40-band Mel spectrogram.
#    - Applies amplitude-to-decibel conversion and per-utterance normalization.

# 4. **Label Speech Activity**
#    - A binary mask is generated: `1` for speech frames, `0` for silence/noise.

# 5. **Segmentation**
#    - The Mel spectrogram and mask are divided into fixed-length segments
#      of `seq_len` frames (10, 25, 50, or 100).
#    - If the utterance is shorter, it is zero-padded.

# 6. **Batching**
#    - The `vad_collate_fn` pads variable-length sequences within a batch.
#    - The `create_vad_dataloaders` function builds DataLoaders for
#      training, validation, and testing splits.

# 👉 In short: this dataloader transforms continuous audio files into
#    small, labeled Mel-spectrogram windows that the model can train on.
# """
# ============================================================

import os
import glob
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import warnings



# ------------------------------------------------------------
# Utility: random noise loader
# ------------------------------------------------------------
def load_random_noise(noise_folder, target_len, fs=16000):
    """Load and trim/pad a random noise waveform to match target length."""
    if not os.path.exists(noise_folder):
        raise FileNotFoundError(f"Noise folder not found: {noise_folder}")
    files = glob.glob(os.path.join(noise_folder, "*.wav"))
    if not files:
        raise RuntimeError(f"No .wav files found in {noise_folder}")

    path = random.choice(files)
    warnings.filterwarnings("ignore", message=r".*torchaudio\.load_with_torchcodec.*", category=UserWarning)
    noise, sr = torchaudio.load(path)
    if sr != fs:
        noise = torchaudio.functional.resample(noise, sr, fs)

    # Repeat / trim to match target length
    if noise.size(1) < target_len:
        reps = int(torch.ceil(torch.tensor(target_len / noise.size(1))))
        noise = noise.repeat(1, reps)
    return noise[:, :target_len]


# ------------------------------------------------------------
# Dataset: Speech + Silence + Noise (with segmentation)
# ------------------------------------------------------------
class SpeechVADDataset(Dataset):
    def __init__(self, subset, base, noise_folder, fs=16000, snr_db=-10, target_dur=1.5, seq_len=25):
        self.subset = subset
        self.fs = fs
        self.noise_folder = noise_folder
        self.snr_db = snr_db
        self.target_samples = int(target_dur * fs)
        self.seq_len = seq_len

        self.mel = T.MelSpectrogram(
            sample_rate=fs,
            n_fft=400,
            hop_length=160,
            n_mels=40,
            f_min=50,
            f_max=7000,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power")

    def _mix_snr(self, clean, noise, snr_db):
        clean_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean()
        scale = torch.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
        return clean + scale * noise

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Load and preprocess waveform
        path = self.subset[idx]
        warnings.filterwarnings("ignore", message=r".*torchaudio\.load_with_torchcodec.*", category=UserWarning)
        speech_wave, sr = torchaudio.load(path)
        if sr != self.fs:
            speech_wave = torchaudio.functional.resample(speech_wave, sr, self.fs)
        speech_wave = speech_wave[:, :self.target_samples]

        # Randomized silence & noise
        silence = torch.zeros(1, int(0.25 * self.fs))
        waveform = torch.cat([silence, speech_wave, silence], dim=-1)
        waveform = waveform[:, :self.target_samples]
        mask = torch.zeros_like(waveform)
        mask[:, silence.shape[-1]:silence.shape[-1] + speech_wave.shape[-1]] = 1

        noise_wave = load_random_noise(self.noise_folder, waveform.size(1), self.fs)
        snr_here = random.uniform(-20, 20)
        mixed = self._mix_snr(waveform, noise_wave, snr_here)

        # Mel + mask
        mel = self.mel(mixed)
        mel_db = self.to_db(mel)
        mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        mel_norm = mel_norm.squeeze(0).transpose(0, 1)  # [T, 40]
        mask_ds = torch.nn.functional.interpolate(mask.unsqueeze(0), size=mel_norm.size(0), mode="nearest").squeeze()

        # Random segment of seq_len frames
        total_frames = mel_norm.size(0)
        if total_frames > self.seq_len:
            start = random.randint(0, total_frames - self.seq_len)
            mel_seg = mel_norm[start:start + self.seq_len]
            mask_seg = mask_ds[start:start + self.seq_len]
        else:
            pad_T = self.seq_len - total_frames
            mel_seg = torch.cat([mel_norm, torch.zeros(pad_T, mel_norm.size(1))], dim=0)
            mask_seg = torch.cat([mask_ds, torch.zeros(pad_T)], dim=0)

        return mel_seg, mask_seg.long(), mel_seg.size(0)



# ------------------------------------------------------------
# Collate function
# ------------------------------------------------------------
def vad_collate_fn(batch):
    xs, ys, lengths = zip(*batch)
    max_len = max([x.size(0) for x in xs])
    n_mels = xs[0].size(1)
    X_pad = torch.zeros(len(xs), max_len, n_mels)
    Y_pad = torch.full((len(xs), max_len), fill_value=-1, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.size(0)
        X_pad[i, :L, :] = x
        Y_pad[i, :L] = y
    lengths = torch.tensor([x.size(0) for x in xs])
    return X_pad, Y_pad, lengths


# ------------------------------------------------------------
# Dataloader builder (no downloading)
# ------------------------------------------------------------
def create_vad_dataloaders(
    data_root="./SpeechCommands",
    noise_train="./noise/train",
    noise_val="./noise/val",
    noise_test="./noise/test",
    snr_db=-10,
    batch_size=16,
    num_workers=2,
    fs=16000,
    seq_len=25,
):
    """Build train, val, and test dataloaders (segmented)."""
    val_list_path = os.path.join(data_root, "validation_list.txt")
    test_list_path = os.path.join(data_root, "testing_list.txt")

    def load_list(file_path):
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    val_list = set(load_list(val_list_path))
    test_list = set(load_list(test_list_path))

    all_files = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.endswith(".wav"):
                rel_path = os.path.relpath(os.path.join(root, f), data_root)
                rel_path = rel_path.replace("\\", "/")
                all_files.append(rel_path)

    train_files, val_files, test_files = [], [], []
    for rel_path in all_files:
        if rel_path in val_list:
            val_files.append(os.path.join(data_root, rel_path))
        elif rel_path in test_list:
            test_files.append(os.path.join(data_root, rel_path))
        else:
            train_files.append(os.path.join(data_root, rel_path))

    print(f"📊 Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    train_ds = SpeechVADDataset(train_files, None, noise_train, fs=fs, snr_db=snr_db, seq_len=seq_len)
    val_ds = SpeechVADDataset(val_files, None, noise_val, fs=fs, snr_db=snr_db, seq_len=seq_len)
    test_ds = SpeechVADDataset(test_files, None, noise_test, fs=fs, snr_db=snr_db, seq_len=seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=vad_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=vad_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=vad_collate_fn)

    print(f"✅ DataLoaders ready: Train={len(train_dl)} batches, Val={len(val_dl)} batches, Test={len(test_dl)} batches")
    return train_dl, val_dl, test_dl
