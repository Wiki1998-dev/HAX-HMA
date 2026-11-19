#!/usr/bin/env python3
"""Emit the ToyAdmos shap program with 8 samples (one mesh tile; see toyadmos.py)."""
import toyadmos

if __name__ == "__main__":
    toyadmos.main("shap", n=8)
