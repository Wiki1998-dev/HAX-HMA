"""Top-1 accuracy of an MLPerf Tiny int8 TFLite model on prepared inputs.

    python scripts/tflite_eval.py kws|vww [--n N]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TINY = os.path.join(ROOT, "external/tiny/benchmark/training")
MODELS = {
    "kws": (os.path.join(TINY, "keyword_spotting/trained_models/kws_ref_model.tflite"),
            os.path.join(ROOT, "external/kws/features_test.npz")),
    "vww": (os.path.join(TINY, "visual_wake_words/trained_models/vww_96_int8.tflite"),
            os.path.join(ROOT, "external/vww/images.npz")),
}


def load_inputs(task: str) -> tuple:
    d = np.load(MODELS[task][1])
    x = d["x"].astype(np.float32)
    if task == "vww":
        x = x / 255.0
    return x, d["y"]


def tflite_logits(task: str, x: np.ndarray) -> np.ndarray:
    """int8 output logits (pre-softmax) and softmax output for each input."""
    from tflite_runtime.interpreter import Interpreter

    it = Interpreter(model_path=MODELS[task][0])
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, zp = inp["quantization"]
    res = []
    for xi in x:
        q = np.clip(np.round(xi / s + zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q.reshape(inp["shape"]))
        it.invoke()
        res.append(it.get_tensor(out["index"])[0].copy())
    return np.stack(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    x, y = load_inputs(a.task)
    if a.n:
        x, y = x[:a.n], y[:a.n]
    p = tflite_logits(a.task, x).argmax(1)
    print(f"{a.task}: TFLite int8 top-1 {np.mean(p == y):.4f} on {len(y)} inputs")
