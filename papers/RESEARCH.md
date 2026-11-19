# Recent XAI & Hardware Acceleration Research

This file contains links to recent (2022-2025) research papers relevant to the XAI-Heterogeneous project, identified via `lixplore-cli` and search.

## 1. Hardware Acceleration of XAI

### ApproXAI: Energy-Efficient Hardware Acceleration of Explainable AI using Approximate Computing
- **Authors:** Ayesha Siddique, Khaza Anuarul Hoque, et al.
- **Year:** 2025 (IJCNN)
- **Links:** [arXiv:2504.17929](https://arxiv.org/abs/2504.17929) | [IEEE Xplore](https://doi.org/10.1109/IJCNN64981.2025.11227243)
- **Relevance:** Explores using approximate multipliers to accelerate Integrated Gradients and Shapley values on edge devices. Direct parallel to our SNAX/CFU acceleration goals.

### Hardware Acceleration of Explainable AI
- **Authors:** (Reference paper in project docs)
- **Link:** [Link](https://ieeexplore.ieee.org/document/9407114)
- **Relevance:** Maps XAI operations to matrix/convolution for hardware acceleration. Foundations for `src/xai/`.

## 2. Gradient SHAP & RISC-V Optimization

### Toward Attention-based TinyML: A Heterogeneous Accelerated Architecture and Automated Deployment Flow
- **Authors:** Philip Wiese, et al.
- **Year:** 2024
- **Links:** [arXiv:2408.02473](https://arxiv.org/abs/2408.02473) | [ETH Zurich Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/687644)
- **Relevance:** Features a hardware-software co-design coupling RISC-V with specialized accelerators. Relevant for our SNAX cluster integration.

## 3. Formal Verification of Quantized DNNs

### Formal Verification of Quantized Neural Network
- **Authors:** Wahiba Bachiri, et al.
- **Year:** 2024 (AFROS)
- **Links:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11036820) | [ResearchGate](https://www.researchgate.net/publication/385494444_Formal_Verification_of_Quantized_Neural_Network)
- **Relevance:** Verification of QNNs using rational approximation and SMT solvers. Directly applicable to Phase 4 formal verification tasks.

### Scalable Verification of Quantized Neural Networks
- **Authors:** Henzinger et al.
- **Year:** 2021 (AAAI)
- **Link:** [arXiv:2012.06484](https://arxiv.org/abs/2012.06484)
- **Relevance:** Foundational work on verifying QNNs at scale.

## 4. Neuro-Symbolic Reasoning

### An Explainable Neuro-Symbolic Explainable Artificial Intelligence Twin Framework for Trustworthy ZSM in Wireless IoE
- **Authors:** Safayat Bin Hakim, et al.
- **Year:** 2022
- **Links:** [arXiv:2210.06744](https://arxiv.org/abs/2210.06744) | [ResearchGate](https://www.researchgate.net/publication/364334444_An_Explainable_Neuro-Symbolic_Explainable_Artificial_Intelligence_Twin_Framework_for_Trustworthy_ZSM_in_Wireless_IoE)
- **Relevance:** Combines neural networks with symbolic reasoning. Useful for the `src/xai/symbolic/` implementation.
