# Hardware Acceleration of Explainable AI on Heterogeneous Multi-Accelerator Platforms

**Project Report**  
**Date**: April 7, 2026  
**Status**: Phase 1 & 2 Verified  
**Target Platforms**: SNAX (RISC-V + GeMM), CFU Playground (FPGA)

---

## 1. Introduction

As Deep Neural Networks (DNNs) are increasingly deployed in safety-critical edge applications—such as autonomous driving and medical diagnostics—the "black-box" nature of these models becomes a significant barrier to trust and regulatory compliance. Explainable AI (XAI) methods, such as Grad-CAM and SHAP, provide visual and numerical attributions that explain model decisions. However, these methods are computationally expensive, often requiring backward passes and multiple sampling iterations that exceed the capabilities of standard edge processors.

This project implements and optimizes XAI methods on **heterogeneous multi-accelerator platforms**. By leveraging a combination of RISC-V scalar cores and specialized accelerators (GeMM, custom FPGA instructions), we aim to achieve XAI overhead of less than 20% of inference latency while maintaining high accelerator utilization.

## 2. Literature Review & Background

### 2.1 Foundational XAI Algorithms
The current implementation is grounded in two primary XAI formulations:
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept.
- **Gradient SHAP (Shapley Additive Explanations)**: Approximates Shapley values by averaging gradients over multiple random baseline samples, providing a mathematically grounded attribution that satisfies the *completeness* axiom.

### 2.2 Hardware Acceleration State-of-the-Art
Our strategy is informed by the following key research:
- **"Hardware Acceleration of Explainable AI"**: This foundational work demonstrates that XAI operations (like the Grad-CAM backward pass) can be mathematically mapped to matrix multiplication (GeMM) and convolution operations, enabling offloading to specialized hardware.
- **ApproXAI (2025)**: Recent research highlighting the use of approximate computing to reduce the energy footprint of XAI on edge devices. We incorporate this by targeting quantized (INT8) acceleration in Phase 2c.
- **DORY (Deployment of Deep Neural Networks on Tiny Systems)**: Provides the memory management strategy for our 128 kB TCDM (Tightly Coupled Data Memory) scratchpad, ensuring collision-free DMA transfers between DRAM and the accelerators.

## 3. Strategy & Methodology

### 3.1 Two-Platform Approach
We employ a heterogeneous strategy to explore different acceleration paradigms:
1.  **SNAX Cluster**: A RISC-V compute cluster featuring Snitch cores and a GeMM accelerator. This platform focuses on data-parallel acceleration of gradient computations.
2.  **CFU Playground**: A framework for adding custom RISC-V instructions to an FPGA-based SoC. This platform focuses on fine-grained SIMD acceleration for saliency accumulation.

### 3.2 Implementation Workflow
Each XAI method follows a rigorous development lifecycle:
1.  **Python Reference**: Implementation in NumPy/PyTorch to establish "golden" ground truth and verify algorithmic correctness.
2.  **Embedded C Kernel**: Porting the reference to restricted C (no dynamic allocation, static buffers) for the RISC-V target.
3.  **Architectural Optimization**: Identifying bottlenecks (e.g., floating-point division, redundant forward passes) and applying hardware-specific fixes like "Backward Hoisting."

### 3.3 Target Model Subset
Until now, we have focused on the **GAP + FC (Global Average Pooling + Fully Connected)** architecture. This subset represents the "decision-making" head of modern classifiers (like ResNet) where Grad-CAM and SHAP are most effective.

## 4. Results & Experimentation

### 4.1 Phase 1: Grad-CAM (Scalar Baseline)
We implemented the Grad-CAM backward pass using a transpose-GeMM pattern on the RISC-V core.
- **Performance**: 6,153 cycles for a 4x4x16 feature map.
- **Bottleneck**: Floating-point division in the normalization step accounted for **68.8%** of total cycles.
- **Verification**: 0/16 BIST errors (100% match with Python reference).

### 4.2 Phase 2: Gradient SHAP (Optimization Cycle)
SHAP represents a significantly higher computational burden due to its sampling nature ($N=16$).

| Phase | Methodology | Cycles | Speedup |
|-------|-------------|--------|---------|
| **2a (Baseline)** | Full pipeline per sample (Interp + Fwd + Bwd + Accum) | 175,510 | 1.0x |
| **2b (Optimized)** | **Backward Hoisting**: Compute gradient once; accumulate only | 58,022 | **3.02x** |

**Experiment Results**:
- **Max Error**: 0.000000 (Exact mathematical equivalence for linear models).
- **Per-Sample Efficiency**: Reduced from 10,969 cycles to 3,626 cycles.

## 5. Observations & Discussion

### 5.1 The "Backward Hoisting" Impact
One of our most significant observations is that for linear model segments (GAP + FC), the gradient is input-independent. By hoisting the backward pass out of the SHAP sampling loop, we eliminated **66.9%** of the computational work (interpolation and forward passes) without any loss in attribution accuracy. This optimization is critical for enabling SHAP on resource-constrained devices.

### 5.2 Memory-Bound Bottlenecks
In the optimized SHAP kernel (Phase 2b), the **Accumulation** stage dominates (**91.6%** of cycles). This confirms that once computational redundancies are removed, XAI performance becomes limited by memory bandwidth (loading input and baseline pairs). This discovery shifts our Phase 3 strategy toward utilizing the SNAX DMA streamers to overlap these memory accesses with the next sample's computation.

### 5.3 FPU Stalls and Instruction Latency
Simulation logs revealed that floating-point division (`fdiv.s`) and loop-carried dependencies in the accumulation loops were the primary sources of stalls. This observation validates our decision to target custom SIMD instructions in the CFU Playground phase to pipeline these operations more effectively.

## 6. Conclusion & Future Work

The project has successfully demonstrated that XAI can be brought to the edge through strategic hardware-software co-design. We have achieved a 3x reduction in SHAP latency through architectural optimization alone. 

**Upcoming Phases**:
- **Phase 3 (CFU)**: Mapping the 91.6% accumulation bottleneck to custom FPGA instructions.
- **Phase 4 (Formal & Symbolic)**: Implementing ILP-based verification to ensure the stability and faithfulness of the generated explanations.
