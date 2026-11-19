"""Prepare MLPerf Tiny keyword-spotting and visual-wake-words inputs.

KWS: Google Speech Commands v0.02 test set (external/speech_commands_test/),
MFCC features exactly as the MLPerf reference (get_dataset.py, feature_type
'mfcc'): each clip divided by its maximum sample, 1 s at 16 kHz, Hann STFT 480 / 320 (fft 512), magnitude, 40 mel bins
20-4000 Hz (tf.signal.linear_to_mel_weight_matrix), log(x + 1e-6), DCT-II
scaled by 1/sqrt(2 * 40) (tf.signal.mfccs_from_log_mel_spectrograms), first
10 coefficients -> (49, 10). Re-implemented in NumPy (no TensorFlow here).

VWW: the 1,000 COCO val2017 images of the MLPerf vww01 evaluation set
(external/vww01_coco/, labels from vww01/y_labels.csv), resized to 96x96
RGB with bilinear interpolation (full image, no crop).

Writes external/kws/features_test.npz and external/vww/images.npz.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXT = os.path.join(ROOT, "external")
KWS_LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes",
              "_silence_", "_unknown_"]


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log1p(f / 700.0)


def mel_matrix(n_mel: int = 40, n_spec: int = 257, sr: int = 16000,
               lo: float = 20.0, hi: float = 4000.0) -> np.ndarray:
    """tf.signal.linear_to_mel_weight_matrix (float64), (n_spec, n_mel)."""
    nyq = sr / 2.0
    lin = np.linspace(0.0, nyq, n_spec)[1:]
    spec_mel = _hz_to_mel(lin)[:, None]
    edges = np.linspace(_hz_to_mel(np.array(lo)), _hz_to_mel(np.array(hi)), n_mel + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    lower_slopes = (spec_mel - lower) / (center - lower)
    upper_slopes = (upper - spec_mel) / (upper - center)
    w = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.pad(w, [[1, 0], [0, 0]])


def mfcc(wave: np.ndarray, mel: np.ndarray) -> np.ndarray:
    """wave: float32 [-1, 1), 16000 samples -> (49, 10) float32 MFCCs."""
    frame, step, nfft = 480, 320, 512
    n = 1 + (wave.size - frame) // step
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)   # periodic Hann
    frames = np.stack([wave[i * step:i * step + frame] for i in range(n)]) * win
    spec = np.abs(np.fft.rfft(frames, nfft)).astype(np.float32)
    logmel = np.log(spec @ mel.astype(np.float32) + 1e-6)
    k = np.arange(40)
    dct = 2.0 * np.cos(np.pi * k[None, :] * (2 * k[:, None] + 1) / 80.0)   # [n, k]
    out = (logmel @ dct) / np.sqrt(80.0)
    return out[:, :10].astype(np.float32)


def kws(limit: int) -> None:
    from scipy.io import wavfile

    base = os.path.join(EXT, "speech_commands_test")
    mel = mel_matrix()
    feats, labels, names = [], [], []
    for li, lab in enumerate(KWS_LABELS):
        files = sorted(f for f in os.listdir(os.path.join(base, lab)) if f.endswith(".wav"))
        for f in files:
            sr, w = wavfile.read(os.path.join(base, lab, f))
            assert sr == 16000 and w.dtype == np.int16
            w = w.astype(np.float32)
            w = w / w.max() if w.max() > 0 else w     # reference: x / reduce_max(x)
            w = np.pad(w, (0, max(0, 16000 - w.size)))[:16000]
            feats.append(mfcc(w, mel))
            labels.append(li)
            names.append(f"{lab}/{f}")
    feats, labels = np.stack(feats), np.array(labels)
    if limit:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(labels), limit, replace=False))
        feats, labels, names = feats[keep], labels[keep], [names[i] for i in keep]
    os.makedirs(os.path.join(EXT, "kws"), exist_ok=True)
    np.savez_compressed(os.path.join(EXT, "kws/features_test.npz"), x=feats, y=labels,
                        names=np.array(names))
    print(f"kws: {len(labels)} clips, features {feats.shape}")


def vww() -> None:
    from PIL import Image

    lab_csv = os.path.join(EXT, "tiny/benchmark/evaluation/datasets/vww01/y_labels.csv")
    imgs, labels, ids = [], [], []
    with open(lab_csv) as f:
        for row in csv.reader(f):
            iid = row[0].split(".")[0]
            im = Image.open(os.path.join(EXT, "vww01_coco", iid + ".jpg")).convert("RGB")
            imgs.append(np.asarray(im.resize((96, 96), Image.BILINEAR), dtype=np.uint8))
            labels.append(int(row[2]))
            ids.append(iid)
    os.makedirs(os.path.join(EXT, "vww"), exist_ok=True)
    np.savez_compressed(os.path.join(EXT, "vww/images.npz"), x=np.stack(imgs),
                        y=np.array(labels), ids=np.array(ids))
    print(f"vww: {len(labels)} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kws-limit", type=int, default=0, help="random subset (0 = all)")
    a = ap.parse_args()
    kws(a.kws_limit)
    vww()
