#!/usr/bin/env python3
"""ToyAdmos (MLPerf Tiny anomaly detection) data pipeline and reference check.

Reproduces the MLPerf Tiny feature extraction
(external/tiny/benchmark/training/anomaly_detection/common.py:
128-bin log-mel, n_fft 1024, hop 512, frames 50..250, 5-frame windows ->
196 vectors x 640 features per clip) on the DCASE 2020 ToyCar test clips,
caches the features, and scores every clip with the int8 reference model
(ad01_int8.tflite): anomaly score = mean over vectors of the MSE between the
dequantised input and output, AUC / pAUC (max_fpr 0.1) per machine ID.

Usage:
    python scripts/toyadmos_data.py            # features (cached) + AUC check
"""

from __future__ import annotations

import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "external")
TEST_DIR = os.path.join(EXT, "toycar", "ToyCar", "test")
CACHE = os.path.join(EXT, "toycar", "features_test.npz")
MODEL = os.path.join(EXT, "tiny", "benchmark", "training", "anomaly_detection",
                     "trained_models", "ad01_int8.tflite")

N_MELS, FRAMES, N_FFT, HOP, POWER = 128, 5, 1024, 512, 2.0


def file_to_vectors(path: str) -> np.ndarray:
    """MLPerf Tiny AD feature extraction for one clip -> (196, 640) float32."""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=False)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, power=POWER)
    logmel = 20.0 / POWER * np.log10(mel + sys.float_info.epsilon)
    logmel = logmel[:, 50:250]
    n = logmel.shape[1] - FRAMES + 1
    vec = np.zeros((n, N_MELS * FRAMES), dtype=np.float32)
    for t in range(FRAMES):
        vec[:, N_MELS * t: N_MELS * (t + 1)] = logmel[:, t: t + n].T
    return vec


def load_features() -> Dict[str, np.ndarray]:
    """Return cached features: names, labels (1 = anomaly), ids, vectors."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.wav")))
    if not files:
        raise FileNotFoundError(f"no clips in {TEST_DIR}")
    with ProcessPoolExecutor() as ex:
        vecs = list(ex.map(file_to_vectors, files, chunksize=16))
    names = np.array([os.path.basename(f) for f in files])
    labels = np.array([1 if n.startswith("anomaly") else 0 for n in names], dtype=np.int8)
    ids = np.array([re.search(r"id_(\d+)", n).group(1) for n in names])
    out = {"names": names, "labels": labels, "ids": ids, "vectors": np.stack(vecs)}
    np.savez_compressed(CACHE, **out)
    return out


TRAIN_DIR = os.path.join(EXT, "toycar", "ToyCar", "train")
BG_CACHE = os.path.join(EXT, "toycar", "features_train_subset.npz")


def load_background() -> np.ndarray:
    """Feature vectors of the extracted normal training subset (every 40th
    training clip): (n_clips * 196, 640) float32. Used for calibration and
    as the baseline / background distribution of the attributions."""
    if os.path.exists(BG_CACHE):
        return np.load(BG_CACHE)["vectors"]
    files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.wav")))
    if not files:
        raise FileNotFoundError(f"no clips in {TRAIN_DIR}")
    with ProcessPoolExecutor() as ex:
        vecs = np.concatenate(list(ex.map(file_to_vectors, files, chunksize=8)))
    np.savez_compressed(BG_CACHE, vectors=vecs)
    return vecs


def tflite_scores(vectors: np.ndarray) -> np.ndarray:
    """Anomaly score per clip with the int8 reference model."""
    from tflite_runtime.interpreter import Interpreter

    it = Interpreter(model_path=MODEL)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = i["quantization"]
    so, zo = o["quantization"]
    scores = np.empty(len(vectors))
    for c, clip in enumerate(vectors):
        q = np.clip(np.round(clip / si + zi), -128, 127).astype(np.int8)
        xd = (q.astype(np.float32) - zi) * si
        err = np.empty(len(clip))
        for v in range(len(clip)):
            it.set_tensor(i["index"], q[v:v + 1])
            it.invoke()
            y = (it.get_tensor(o["index"])[0].astype(np.float32) - zo) * so
            err[v] = np.mean((xd[v] - y) ** 2)
        scores[c] = err.mean()
    return scores


def auc_per_id(labels: np.ndarray, ids: np.ndarray,
               scores: np.ndarray) -> List[Tuple[str, float, float]]:
    """AUC and pAUC (max_fpr 0.1) per machine ID, as in the MLPerf test."""
    from sklearn import metrics

    out = []
    for m in sorted(set(ids)):
        k = ids == m
        out.append((m, metrics.roc_auc_score(labels[k], scores[k]),
                    metrics.roc_auc_score(labels[k], scores[k], max_fpr=0.1)))
    return out


def main() -> None:
    d = load_features()
    print(f"{len(d['names'])} clips ({int(d['labels'].sum())} anomalous), "
          f"vectors {d['vectors'].shape}")
    s = tflite_scores(d["vectors"])
    np.save(os.path.join(EXT, "toycar", "tflite_int8_scores.npy"), s)
    res = auc_per_id(d["labels"], d["ids"], s)
    for m, a, p in res:
        print(f"  id_{m}: AUC {a:.4f}  pAUC {p:.4f}")
    print(f"  mean : AUC {np.mean([r[1] for r in res]):.4f}  "
          f"pAUC {np.mean([r[2] for r in res]):.4f}")


if __name__ == "__main__":
    main()
