# Proposed API and behavior contract

Status: design only. The blend equation is accepted; names and defaults below await [decision review](DECISIONS.md). Examples are target interfaces, not runnable current functionality. All learned activation parameters represent blend coefficients in `[0, 1]`, not exponents. No aliases for the superseded power proposal will be added.

## Public surface

Limit initial top-level exports to `ExpoActivation`, `ExpoMLP`, `ExpoRegressor`, and `ExpoClassifier`, plus package version metadata. Persistence is exposed as estimator methods, not another lifecycle API.

### PyTorch modules

```python
ExpoActivation(
    num_features=None, *, blend_mode="per_neuron",
    blend_init=0.5, trainable=True, feature_dim=-1,
)

ExpoMLP(
    in_features, out_features, *, hidden_dims=(64, 64),
    blend_mode="per_neuron", blend_init=0.5,
    trainable_blend=True, normalization="layer",
)
```

- `ExpoActivation` requires a positive `num_features` in per-neuron mode. In per-layer mode require `num_features=None`; `feature_dim` has no effect there.
- Per-neuron mode validates that the selected input axis matches `num_features`, normalizes negative axis indices, and broadcasts across all other axes. At least one input dimension is required.
- `.blend_weight` returns the effective coefficient tensor `a`; in trainable mode it remains connected to autograd. Expose no setter. Inspection must not mutate it in place.
- Both modules follow `nn.Module` behavior for `.to(...)`, `.train()`, `.eval()`, `parameters()`, and `state_dict()`; no hidden CPU copies or NumPy conversions in forward.
- `ExpoMLP` accepts only `(batch, in_features)` tensors and returns `(batch, out_features)`. The activation is more generally reusable without promising image/sequence estimators.
- `hidden_dims` must be a nonempty tuple of positive widths; LayerNorm requires every width to be at least two. No output activation, implicit flattening, residual path, or dropout.
- The module assumes caller-managed input scaling, dtype, device, optimizer, and task loss. Float32 is the primary execution dtype; CPU float64 supports mathematical tests.

### Estimator configuration

Common constructor parameters are explicit keyword arguments with the following proposed defaults:

| Parameter | Default | Contract |
| --- | --- | --- |
| `hidden_dims` | `(64, 64)` | Same architecture contract as ExpoMLP. |
| `blend_mode` | `"per_neuron"` | `"per_neuron"` or proposed experimental `"per_layer"`. |
| `blend_init` | `0.5` | Scalar coefficient; `0 < blend_init < 1` when trainable, `[0, 1]` when fixed. |
| `trainable_blend` | `True` | False permits fixed endpoints. |
| `normalization` | `"layer"` | `"layer"` or `"none"`; provisional default. |
| `standardize` | `True` | Training-only feature StandardScaler. |
| `lr` | `1e-3` | Positive finite main learning rate. |
| `blend_lr` | `None` | None means main learning rate; otherwise positive finite. |
| `weight_decay` | `0.0` | Nonnegative finite; linear weight matrices only. |
| `batch_size` | `128` | Positive integer; retain a final incomplete batch. |
| `epochs` | `100` | Positive integer maximum. |
| `shuffle` | `True` | Shuffle training examples each epoch. |
| `early_stopping` | `False` | Requires a validation partition when enabled. |
| `validation_fraction` | `0.1` | Used only to create an internal holdout for early stopping. |
| `patience` | `15` | Positive integer epochs without sufficient improvement. |
| `min_delta` | `0.0` | Nonnegative absolute improvement in validation loss. |
| `max_grad_norm` | `None` | None disables clipping; otherwise positive finite. |
| `device` | `"cpu"` | `"cpu"`, `"cuda"`, `"cuda:N"`, or `"auto"`. |
| `random_state` | `None` | Integer seed or None; no RandomState/Generator objects initially. |
| `verbose` | `0` | 0 is quiet; 1 reports epoch summaries. |

`ExpoRegressor` additionally accepts `target_standardize=False`. `ExpoClassifier` does not accept that parameter. The first version fixes AdamW and task-appropriate losses; custom losses, optimizer registries, and schedulers are omitted.

### Methods and fitted attributes

```python
est.fit(X, y, *, validation_data=None)  # returns self
est.predict(X)                        # NumPy array
est.score(X, y)                       # R2 or accuracy
est.get_params(deep=True)
est.set_params(**params)
est.get_blend_weights()               # copied NumPy arrays, one per hidden layer
est.save(path)                        # inference snapshot, after fitting
ExpoRegressor.load(path, *, device="cpu")
ExpoClassifier.load(path, *, device="cpu")

clf.predict_proba(X)                  # columns correspond to classes_
```

Fitted attributes: `model_`, `n_features_in_`, `history_`, `n_iter_`, `device_`, `feature_scaler_` (None when disabled), and `best_epoch_` (None when early stopping is disabled). Regression adds `n_outputs_`, `target_scaler_`, and private original-target-rank metadata. Classification adds `classes_`.

Do not create fitted attributes or allocate devices in the estimator constructor. Store constructor arguments unchanged and validate in `fit`. Use BaseEstimator with the appropriate regressor/classifier mixin. Verify cloning, parameter search, and Pipeline integration against the supported scikit-learn version. [Estimator development guide](https://scikit-learn.org/stable/developers/develop.html).

Repeated `fit` starts a fresh model, optimizer, scalers, label encoding, and history; no `warm_start` or `partial_fit`. Prediction before successful fitting raises `NotFittedError`. A failed new fit must not leave an apparently fitted partial model. Changing parameters requires a new fit before prediction; invalidate fitted state on a nonempty `set_params` call and test this behavior.

## Data and task semantics

- Estimators initially accept dense real numeric NumPy arrays or numeric array-like features of shape `(N, F)`, with `N > 0`, `F > 0`. Reject sparse, complex, missing, and infinite values. Convert numeric features to float32 and check finiteness again after conversion/scaling.
- Torch tensors belong to the direct module interface; do not silently detach or transfer estimator inputs. Dataframe-specific feature-name guarantees and automatic categorical encoding are deferred; examples should use NumPy explicitly.
- Regression targets may be `(N,)` or `(N, K)` with `K >= 1`. Preserve that rank in predictions, including the distinction between `(N,)` and `(N, 1)`.
- Regression uses MSE averaged over batch elements and outputs. `score` is scikit-learn R2 in original target units with uniform averaging over outputs.
- Classification targets are a one-dimensional array of integer or string labels of one consistent type, with at least two training classes. Reject continuous targets, mixed object labels, missing labels, and multilabel/multioutput targets.
- Use `K` logits and cross-entropy for all classification tasks, including binary classification. Convert logits with softmax only at inference. `predict_proba` returns `(N, K)` in `classes_` order; `predict` maps argmax back to labels, choosing the first class in that order on a tie. Score is accuracy.
- Inputs, targets, and validation arrays must have consistent sample/feature/output shapes. Reshape regression targets deliberately once; never let a loss broadcast them implicitly.
- Validation classification labels must be in the fitted class set. Standardization never applies to labels.
- Prediction keeps input row order, works with one sample, and processes bounded batches under evaluation and inference mode. Reject zero-row prediction inputs in the initial interface.

## Validation, scaling, and early stopping

1. Validate constructor settings and raw data before building the model.
2. Explicit `validation_data=(X_val, y_val)` takes precedence over internal holdout creation. Use it for monitoring even if early stopping is off; do not split again.
3. Without explicit validation data, create a holdout only when early stopping is on. Require `0 < validation_fraction < 1`; use a seeded random split for regression and a stratified split for classification. Fail clearly if class counts cannot support the requested split.
4. Split before fitting either feature or target scalers. Never learn scaler statistics from validation or test data. With no holdout, fit scalers on all supplied training rows.
5. For chronological/grouped workloads, the caller must provide an appropriate explicit holdout. The default random split does not handle these structures. A preceding Pipeline transformer may already have seen an internal holdout; use estimator-owned scaling for a strict internal validation boundary, or split and transform outside the Pipeline explicitly.
6. Train with mini-batches; compute each epoch's train loss weighted by batch sample count. Compute validation loss in evaluation mode over all validation rows, similarly weighted.
7. Early stopping monitors validation loss (scaled target units when target scaling is on). An improvement is `new_loss < best_loss - min_delta`. The first finite validation result establishes the best epoch. Ties are non-improvements.
8. Stop after `patience` consecutive non-improving epochs. Restore a deep copy of all model parameters/buffers from the best epoch whenever early stopping is enabled, including when the epoch limit is reached. Otherwise retain the last epoch's model.
9. `n_iter_` counts executed epochs, not the best epoch. `history_` retains all executed epochs; best-epoch indices are one-based.

Each history entry contains `epoch`, sample-weighted `train_loss`, `val_loss` or None, elapsed epoch seconds, and per-layer effective blend coefficient min/mean/max. Stored values are detached Python numbers. Full activation/gradient distributions belong in the benchmark runner, not permanent per-batch library history.

## Device and reproducibility policy

`device="auto"` chooses CUDA when available, otherwise CPU. Explicit CUDA requests must succeed or raise; no silent CPU fallback. Record the resolved device in `device_`. GPU installation is controlled by the caller's PyTorch distribution; do not install drivers or substitute a Torch build.

The estimator uses float32 model parameters and batches on the selected device, with source data held on CPU. An integer `random_state` controls splitting, initialization, and batch order. Scope/restore any process-global random state used for initialization; do not permanently change deterministic or backend settings. Same seed does not promise bitwise equality across hardware or library versions. Parallel cross-validation should not oversubscribe one GPU; document `n_jobs=1` for that case.

AMP, automatic compilation, custom device pools, and distributed execution are not accepted estimator parameters in the first release.

## Persistence boundary

An inference snapshot must contain a format version, estimator kind, constructor configuration, model state, fitted scaler arrays, original regression target rank or classification labels, feature/output dimensions, training summary, and producer package/library versions. Encode state as tensors and primitive metadata; avoid serialized estimator objects and arbitrary callable factories.

Save CPU copies without moving the live model. Reconstruct only the built-in ExpoNet types, validate metadata and state shapes, load strictly with an explicit restricted `weights_only=True` path, and honor the caller's load device. Reject mismatched estimator kinds and unknown format versions. Write through a temporary sibling file followed by atomic replacement; do not truncate an existing snapshot on an unsuccessful write.

Restricted loading reduces arbitrary-code execution exposure but is not a sandbox for hostile files and does not prevent all resource-exhaustion risks. Do not use unrestricted pickle fallback. [PyTorch serialization notes](https://docs.pytorch.org/docs/2.9/notes/serialization.html#torch-load-with-weights-only-true).

Snapshot loading restores inference, blend-weight inspection, fitted metadata, and history. It does not restore optimizer/RNG state or promise continuation of training. Calling `fit` on a loaded estimator starts a fresh fit. Exact resume is deferred.
