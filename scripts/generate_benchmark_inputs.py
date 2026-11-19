#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_resnet8_input() -> None:
    import torchvision

    print("[1/3] Downloading CIFAR-10 test set …")
    ds = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True)

    target_class = 1  # automobile
    arr = None
    for img, label in ds:
        if label == target_class:
            arr = np.array(img, dtype=np.float32) / 255.0
            break
    if arr is None:
        raise RuntimeError("No automobile image found in CIFAR-10 test set")

    assert arr.shape == (32, 32, 3)
    out = OUT_DIR / "resnet8_input.npy"
    np.save(out, arr)
    print(f"  ✓ Saved {out}  shape={arr.shape}  dtype={arr.dtype}")
    print(f"    pixel range: [{arr.min():.4f}, {arr.max():.4f}]")


def generate_mobilebert_input() -> None:
    from transformers import MobileBertTokenizer, MobileBertModel
    import torch

    print("[2/3] Loading google/mobilebert-uncased …")
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    model = MobileBertModel.from_pretrained("google/mobilebert-uncased")
    model.eval()

    sentence = "The machine is operating normally."
    enc = tokenizer(
        sentence,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        # Get embeddings from the embedding layer
        input_embeds = model.embeddings(enc["input_ids"], enc.get("token_type_ids"))

    arr = input_embeds[0].detach().cpu().numpy().astype(np.float32)

    # MobileBERT uses 512-dimensional embeddings, not 128
    # The shape should be (seq_length, embedding_dim) = (32, 512)
    expected_shape = (32, 512)
    assert arr.shape == expected_shape, f"Unexpected shape {arr.shape}, expected {expected_shape}"

    out = OUT_DIR / "mobilebert_input.npy"
    np.save(out, arr)
    print(f"  ✓ Saved {out}  shape={arr.shape}  dtype={arr.dtype}")
    print(f"    value range: [{arr.min():.4f}, {arr.max():.4f}]")


def generate_toyadmos_input() -> None:
    import urllib.request
    import librosa

    print("[3/3] Downloading sample audio …")
    url = "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"
    wav_path = "/tmp/sample_audio.wav"

    if not os.path.exists(wav_path):
        urllib.request.urlretrieve(url, wav_path)
        print(f"  Downloaded → {wav_path}")

    y, sr = librosa.load(wav_path, sr=16000)
    print(f"  Audio: {len(y)} samples @ {sr} Hz ({len(y) / sr:.1f}s)")

    n_mels = 64
    n_frames_needed = 10
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=512,
        n_fft=1024,
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    print(f"  Mel spectrogram: {log_S.shape} (mels × frames)")

    if log_S.shape[1] < n_frames_needed:
        pad_width = n_frames_needed - log_S.shape[1]
        log_S = np.pad(log_S, ((0, 0), (0, pad_width)), mode="edge")

    patch = log_S[:, :n_frames_needed]
    arr = patch.T.flatten().astype(np.float32)

    assert arr.shape == (640,), f"Unexpected shape {arr.shape}"
    out = OUT_DIR / "toyadmos_input.npy"
    np.save(out, arr)
    print(f"  ✓ Saved {out}  shape={arr.shape}  dtype={arr.dtype}")
    print(f"    value range: [{arr.min():.4f}, {arr.max():.4f}]")


if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    generate_resnet8_input()
    print()
    generate_mobilebert_input()
    print()
    generate_toyadmos_input()
    print(f"\n{'=' * 60}")
    print("All three benchmark inputs generated successfully.")
    for f in sorted(OUT_DIR.glob("*.npy")):
        a = np.load(f)
        print(f"  {f.name:30s} shape={str(a.shape):15s} dtype={a.dtype}")