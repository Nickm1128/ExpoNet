# Initial controlled evaluation

Measured: 2026-08-30. Scope: CPU float32, five paired seeds, four small numeric workloads, and the P1-P5 implementation. This report distinguishes implementation correctness from empirical results; it is not evidence that any activation is universally better.

## Protocol and reproducibility

The complete runner is [initial_evaluation.py](../benchmarks/initial_evaluation.py). It ran all seven activation variants in [VALIDATION.md](VALIDATION.md): native ReLU, native squared ReLU, fixed ExpoActivation at `a=0`, `0.5`, and `1`, learned per layer, and learned per neuron. Every variant used the same `(16,)` hidden width, Xavier initialization seed, batch size 128, learning rate 0.01, 35 epochs, train-only feature scaling, and target scaling for regression. Fixed Expo `a=0`/`a=1` exactly matched the native endpoint metrics because the matched blocks and seeds are identical.

For each workload/seed, a paired 64/16/20 train/validation/test split was created before any scaling. Classification splits were stratified. The validation partition was monitored but did not select hyperparameters; test partitions were held out from the fixed protocol. One CPU thread was used. Timing is end-to-end estimator/model training time per run, including preprocessing and model setup; it is not isolated kernel timing.

```powershell
.\.venv-cuda\Scripts\python.exe -B -m benchmarks.initial_evaluation --device cpu --epochs 35 --seeds 11,23,37,53,71 --output reports\initial-evaluation-20260830.json
```

The ignored raw report at `reports/initial-evaluation-20260830.json` retains all 280 per-seed configuration, split, metric, runtime, parameter-count, blend, and failure records. All 280 attempted runs completed.

Environment: Windows 11; Python 3.12.10; PyTorch 2.11.0+cu128; NumPy 2.5.2; scikit-learn 1.9.0; CPU execution with CUDA available but unused for this controlled matrix.

## Workloads and provenance

| Workload | Task | Size | Source and license note | SHA-256 of features + targets |
| --- | --- | ---: | --- | --- |
| Synthetic regression | Regression | 240 × 8 | `make_regression`, seed 20260830; generated locally, no external dataset license | `794ced05c6cc2a32dfed31e946ce2a56f29151289c1249d24cf34169e7127cec` |
| Diabetes | Regression | 442 × 10 | Bundled scikit-learn Diabetes dataset (Efron et al., 2004); scikit-learn is BSD-3-Clause and its bundled description names no separate source-data license | `dfe8bf10292413576e3fd58c22ce77ad4ab87ab7db3990a2780b756ac685c9dd` |
| Synthetic multiclass | Classification | 240 × 8 | `make_classification`, seed 20260830; generated locally, no external dataset license | `b75fc7ad38a9b50ab3100df4a9a1f337b50122b5da2017fe5cfc73864d33290e` |
| Iris | Classification | 150 × 4 | Bundled scikit-learn Iris dataset (Fisher, 1936); scikit-learn is BSD-3-Clause and its bundled description names no separate source-data license | `879b81724528964779983358162a724401de00141f8cb9c922d27f908396b8c7` |

## Aggregate held-out results

Values are mean ± sample standard deviation over the five paired seeds. Regression cells are `RMSE / MAE / R2`; lower RMSE/MAE and higher R2 are better. Classification cells are `accuracy / log loss`; higher accuracy and lower log loss are better. The raw report contains the corresponding per-seed values and the native squared-ReLU rows.

### Regression

| Workload | Hidden norm | ReLU / fixed `a=0` | Fixed `a=0.5` | Fixed `a=1` | Learned per layer | Learned per neuron |
| --- | --- | --- | --- | --- | --- |
| Synthetic | None | 22.52 ± 1.53 / 18.13 ± 1.61 / 0.970 ± 0.007 | 25.78 ± 4.11 / 20.81 ± 3.15 / 0.959 ± 0.018 | 39.43 ± 5.51 / 31.22 ± 4.30 / 0.905 ± 0.034 | 23.94 ± 3.42 / 19.26 ± 2.76 / 0.965 ± 0.014 | 24.77 ± 3.81 / 19.92 ± 2.96 / 0.962 ± 0.016 |
| Synthetic | LayerNorm | 45.26 ± 6.44 / 34.47 ± 4.53 / 0.878 ± 0.042 | 46.75 ± 3.03 / 37.49 ± 1.36 / 0.872 ± 0.025 | 53.62 ± 5.37 / 42.37 ± 4.39 / 0.831 ± 0.039 | 46.31 ± 3.12 / 37.07 ± 1.37 / 0.874 ± 0.026 | 46.47 ± 3.12 / 37.24 ± 1.37 / 0.873 ± 0.025 |
| Diabetes | None | 55.52 ± 2.27 / 44.16 ± 1.81 / 0.499 ± 0.089 | 58.89 ± 4.76 / 46.88 ± 2.33 / 0.435 ± 0.134 | 61.66 ± 6.70 / 49.47 ± 3.61 / 0.378 ± 0.178 | 58.30 ± 4.29 / 46.42 ± 2.18 / 0.446 ± 0.126 | 58.67 ± 4.49 / 46.59 ± 2.17 / 0.440 ± 0.127 |
| Diabetes | LayerNorm | 56.28 ± 2.48 / 44.62 ± 1.70 / 0.489 ± 0.068 | 56.59 ± 1.98 / 44.79 ± 1.58 / 0.484 ± 0.064 | 57.37 ± 1.88 / 45.20 ± 2.23 / 0.468 ± 0.080 | 56.49 ± 2.00 / 44.73 ± 1.53 / 0.485 ± 0.063 | 56.47 ± 2.05 / 44.69 ± 1.61 / 0.486 ± 0.064 |

### Classification

| Workload | Hidden norm | ReLU / fixed `a=0` | Fixed `a=0.5` | Fixed `a=1` | Learned per layer | Learned per neuron |
| --- | --- | --- | --- | --- | --- |
| Synthetic | None | 0.867 ± 0.056 / 0.325 ± 0.060 | 0.858 ± 0.040 / 0.348 ± 0.086 | 0.854 ± 0.047 / 0.389 ± 0.131 | 0.858 ± 0.040 / 0.351 ± 0.084 | 0.863 ± 0.046 / 0.347 ± 0.082 |
| Synthetic | LayerNorm | 0.888 ± 0.038 / 0.309 ± 0.057 | 0.879 ± 0.034 / 0.360 ± 0.107 | 0.875 ± 0.015 / 0.434 ± 0.155 | 0.879 ± 0.034 / 0.368 ± 0.114 | 0.879 ± 0.034 / 0.372 ± 0.118 |
| Iris | None | 0.887 ± 0.096 / 0.291 ± 0.111 | 0.913 ± 0.077 / 0.277 ± 0.113 | 0.900 ± 0.082 / 0.275 ± 0.120 | 0.913 ± 0.077 / 0.277 ± 0.113 | 0.913 ± 0.077 / 0.274 ± 0.113 |
| Iris | LayerNorm | 0.913 ± 0.069 / 0.235 ± 0.139 | 0.947 ± 0.056 / 0.164 ± 0.115 | 0.947 ± 0.056 / 0.135 ± 0.102 | 0.947 ± 0.056 / 0.162 ± 0.116 | 0.947 ± 0.056 / 0.161 ± 0.114 |

## Paired learned-vs-ReLU differences

Each entry is the mean ± sample standard deviation of per-seed `learned − native ReLU` primary-metric differences. Positive RMSE is worse; positive accuracy is better.

| Workload | Hidden norm | Per layer | Per neuron |
| --- | --- | --- | --- |
| Synthetic regression | None | RMSE +1.41 ± 2.44 | RMSE +2.25 ± 2.78 |
| Synthetic regression | LayerNorm | RMSE +1.05 ± 4.67 | RMSE +1.21 ± 4.71 |
| Diabetes | None | RMSE +2.78 ± 2.99 | RMSE +3.15 ± 3.33 |
| Diabetes | LayerNorm | RMSE +0.20 ± 0.58 | RMSE +0.19 ± 0.50 |
| Synthetic multiclass | None | Accuracy −0.008 ± 0.024 | Accuracy −0.004 ± 0.023 |
| Synthetic multiclass | LayerNorm | Accuracy −0.008 ± 0.019 | Accuracy −0.008 ± 0.019 |
| Iris | None | Accuracy +0.027 ± 0.028 | Accuracy +0.027 ± 0.028 |
| Iris | LayerNorm | Accuracy +0.033 ± 0.041 | Accuracy +0.033 ± 0.041 |

The learned variants added one (per-layer) or sixteen (per-neuron) trainable parameters. On this one-thread CPU run, their mean training time was typically 9–28% above the native ReLU reference; a single synthetic-ReLU timing outlier is retained in the raw report rather than discarded.

## Coefficient behavior and decision

Learned coefficients moved from 0.5 without endpoint saturation. Across the five seeds, per-layer means ranged from 0.409–0.581; per-neuron means ranged from 0.463–0.535, with individual values spanning 0.378–0.687. No learned coefficient was within 0.01 of either endpoint. Regression workloads shifted coefficients below 0.5, whereas the classification workloads were nearer or above 0.5; this is descriptive activation behavior, not feature importance.

The evidence does not support changing `blend_init=0.5`, the primary per-neuron option, or the provisional LayerNorm default. No-normalization performed better on both regression workloads, while LayerNorm performed better on both classification workloads; the learned blend did not consistently improve held-out results over ReLU, and the five-seed dispersion is material. Keep both normalization modes exposed, retain ReLU as the practical runtime baseline, and defer a broader default change until larger and more varied controlled evaluations.

These runs establish implementation behavior and numerical stability for the tested settings. They do not establish predictive superiority, a universal activation choice, or GPU end-to-end performance.
