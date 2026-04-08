"""Neuro-symbolic rule extraction from XAI saliency maps.

Extracts shallow decision trees from feature saliency, then exports them
as C headers for embedded execution on RISC-V (no floating-point required).

Connects Phase 1 (Grad-CAM saliency) → Phase 4 (symbolic rules):
  Saliency tells us WHICH features matter →
  Decision tree tells us WHAT thresholds matter →
  Rules can be audited, formally verified, and run on tiny hardware.
"""
