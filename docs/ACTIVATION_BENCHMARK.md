# P1.04 activation-overhead measurement

Measured: 2026-08-30. Scope: Windows CPU float32 only. CUDA was unavailable in the recorded Python/PyTorch environment and was not tested.

This is an early timing review, not an end-to-end model benchmark. It compares native ReLU, native squared ReLU, plain sine, and the real trainable `ExpoActivation`. Sine is a timing reference only and is not a supported ExpoNet activation. No performance threshold was agreed before measurement, so these results do not establish a speed pass, speed advantage, or release claim.

## Environment and protocol

- Platform: Windows 10, build 26200.
- Processor: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel.
- Python 3.11.9; PyTorch 2.7.1+cu118; NumPy 2.4.6; scikit-learn 1.4.2.
- ExpoNet import: `C:\Users\milin\Documents\ExpoNet\src\exponet\__init__.py`.
- Device: CPU; `torch.cuda.is_available()` was `False`.
- Dtype: float32. PyTorch intra-operation threads: 1.
- Deterministic seed: 20260830.
- Shapes: batch sizes 1 and 128; widths 64 and 256.
- Modes: inference and forward plus backward.
- Dense blocks: `Linear(width, width)` followed by either Identity or LayerNorm, then the selected activation.
- Warmup: 20 explicit iterations for every case.
- Timing: `torch.utils.benchmark.Timer.blocked_autorange` with a 0.5-second minimum per case. Setup, module construction, deterministic input creation, and transfers were outside timed regions. Training timings consistently cleared gradients inside the timed operation.
- Result format below: median microseconds `[IQR microseconds]`. The final 100-case run had no result with IQR greater than 25% of its median.

Exact final command:

```powershell
python -B -m benchmarks.benchmark_activation --device cpu --num-threads 1 --warmup 20 --min-run-time 0.5 --output "$env:TEMP\exponet-activation-benchmark-cpu-20260830-final.json"
```

Result: `measurements=100`.

## Coefficient mapping alone

This isolates `sigmoid(theta)` on the width-sized coefficient vector.

| Width | Inference | Forward + backward |
| ---: | ---: | ---: |
| 64 | 4.96 [0.57] | 32.30 [6.53] |
| 256 | 4.76 [0.38] | 30.91 [3.59] |

## Isolated activations

| Mode | Batch | Width | ReLU | Squared ReLU | Sine | Learned blend |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Inference | 1 | 64 | 5.78 [0.51] | 7.55 [0.84] | 6.04 [0.84] | 18.19 [1.83] |
| Inference | 1 | 256 | 5.77 [0.47] | 7.29 [0.65] | 6.43 [1.17] | 18.48 [1.98] |
| Inference | 128 | 64 | 6.69 [0.72] | 8.67 [0.57] | 9.55 [0.71] | 24.28 [4.98] |
| Inference | 128 | 256 | 8.24 [1.17] | 11.73 [0.65] | 20.72 [1.53] | 31.00 [3.35] |
| Forward + backward | 1 | 64 | 39.20 [4.40] | 51.99 [3.08] | 39.76 [4.47] | 106.97 [7.80] |
| Forward + backward | 1 | 256 | 42.01 [8.48] | 54.78 [6.24] | 39.98 [3.75] | 109.69 [10.86] |
| Forward + backward | 128 | 64 | 43.90 [6.12] | 62.77 [8.92] | 51.91 [9.29] | 137.07 [18.60] |
| Forward + backward | 128 | 256 | 57.66 [6.57] | 84.57 [8.58] | 86.65 [12.34] | 179.00 [38.97] |

The learned blend measured 3.15-3.76 times the isolated ReLU inference median and 2.61-3.12 times its isolated forward/backward median. The separate sigmoid mapping accounted for 15.3-30.2% of the complete learned-activation median across these cases; mapping cost is material but does not explain the full difference.

## Minimal dense blocks without normalization

| Mode | Batch | Width | ReLU | Squared ReLU | Sine | Learned blend |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Inference | 1 | 64 | 15.01 [2.48] | 15.26 [1.03] | 13.83 [0.99] | 31.60 [3.11] |
| Inference | 1 | 256 | 16.28 [1.42] | 17.75 [1.68] | 16.43 [1.35] | 35.41 [3.54] |
| Inference | 128 | 64 | 26.05 [4.13] | 27.76 [2.93] | 29.78 [3.26] | 47.69 [4.96] |
| Inference | 128 | 256 | 159.41 [10.83] | 165.09 [13.16] | 172.46 [12.47] | 199.87 [16.79] |
| Forward + backward | 1 | 64 | 92.47 [9.92] | 106.03 [7.87] | 92.24 [9.00] | 162.77 [19.63] |
| Forward + backward | 1 | 256 | 110.61 [10.30] | 131.27 [27.26] | 113.71 [11.30] | 187.50 [19.53] |
| Forward + backward | 128 | 64 | 130.54 [15.69] | 157.69 [30.09] | 138.68 [14.95] | 226.45 [22.87] |
| Forward + backward | 128 | 256 | 556.20 [47.90] | 608.06 [53.92] | 596.14 [56.23] | 726.66 [112.11] |

Against the matched ReLU block, the learned-blend block measured 1.25-2.18 times the inference median and 1.31-1.76 times the forward/backward median.

## Minimal dense blocks with LayerNorm

| Mode | Batch | Width | ReLU | Squared ReLU | Sine | Learned blend |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Inference | 1 | 64 | 20.66 [2.14] | 23.73 [3.50] | 20.71 [4.27] | 39.98 [5.58] |
| Inference | 1 | 256 | 23.28 [3.60] | 25.83 [3.41] | 22.37 [2.67] | 44.44 [5.33] |
| Inference | 128 | 64 | 39.71 [5.13] | 42.08 [3.46] | 42.63 [4.84] | 65.76 [6.51] |
| Inference | 128 | 256 | 179.67 [22.10] | 198.89 [32.91] | 197.01 [25.83] | 214.60 [18.66] |
| Forward + backward | 1 | 64 | 129.10 [12.24] | 142.10 [20.28] | 130.95 [10.48] | 206.74 [45.58] |
| Forward + backward | 1 | 256 | 148.06 [16.33] | 162.52 [11.19] | 147.75 [18.36] | 222.72 [26.07] |
| Forward + backward | 128 | 64 | 184.27 [14.02] | 207.34 [38.62] | 192.04 [15.86] | 288.15 [45.27] |
| Forward + backward | 128 | 256 | 622.65 [52.09] | 638.60 [42.90] | 670.66 [73.71] | 729.20 [52.05] |

Against the matched ReLU plus LayerNorm block, the learned-blend block measured 1.19-1.94 times the inference median and 1.17-1.60 times the forward/backward median.

## Review and limitations

The accepted blend has clear eager-CPU overhead in isolation, especially for small batches where parameter mapping, broadcasting, module dispatch, and additional elementwise operations are poorly amortized. The relative difference narrows in dense blocks and is smallest in the largest LayerNorm case because matrix multiplication and normalization dominate more of the total time. The results do not show a runtime advantage over ReLU, squared ReLU, or sine.

No performance budget has been selected, so this measurement does not create a numerical pass/fail threshold. It does provide an evidence-based baseline for P2: proceed without claiming speed, retain native ReLU as the realistic runtime baseline, and measure complete training workloads before changing defaults or adding optimization complexity.

This run covers one Windows CPU, float32, eager execution, one PyTorch version, one thread, and small dense shapes. It does not cover end-to-end training, multiple thread counts, compilation, other operating systems, other CPUs, reduced precision, or GPU execution. A CUDA invocation was prepared, but the recorded Python/PyTorch environment reported no available CUDA device; no skipped or rejected command counts as CUDA timing evidence.
