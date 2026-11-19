# GeMMX Launch Configuration Overhead

**Finding.** With the stock driver, configuring one GeMMX launch costs ~2,300 cycles, while the accelerator often computes for only tens to a few thousand cycles. Configuration dominated every XAI workload (52% of ToyAdmos IG).

**Cause.** `csrw_ss()` (`sw/snRuntime/src/csr.h`) is a `switch` over all CSR addresses: RISC-V `csrw` needs an immediate address. `set_gemmx_streamer_csr` / `set_gemmx_csr` call it in loops over arrays, so the 75 register writes per launch each pay for a computed jump, loop overhead and array loads.

**Fix (no hardware change).** `gemmx_sched.py` knows every register value of every launch at build time. It generates one function per launch with straight-line `write_csr(<constant>, value)` and, because the CSR-manager registers persist until rewritten (they are latched at start), writes only the registers that changed since the previous launch.

| Mode (14-launch pattern test, all outputs bit-exact) | Config cycles | per launch |
|---|---:|---:|
| Library (`switch` per CSR) | 32,250 | ~2,300 |
| Generated straight-line, all 75 registers | 2,595 | ~185 |
| **Generated delta writes** | **1,080** | **~77** |

| Workload | Library | Delta | Speed-up |
|---|---:|---:|---:|
| ToyAdmos inference | 47,189 | 9,227 | 5.1× |
| ToyAdmos IG / Gradient SHAP (with fused inference) | 181,188 | 91,517 | 2.0× |

**Not possible:** writing launch *n*+1's registers while launch *n* runs hangs the cluster (a CSR write during a run stalls the interface), even though the registers are buffered. With delta writes at ~77 cycles per launch, the remaining gain would have been small.
