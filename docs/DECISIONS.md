# Design decisions

Updated: 2026-08-30. Status: ReLU/squared-ReLU blend and MIT license accepted; remaining defaults and scope are under review.

This is the authority for scope and accepted choices. The design and API documents expand the proposals below; they do not make an unconfirmed proposal binding. Record each decision here before updating dependent documents.

## Confirmed requirements

- Project identity: ExpoNet; implementation built on PyTorch.
- One activation family: `u = max(x, 0); f(x; a) = (1-a)*u + a*u*u`.
- A continuous blend coefficient `a` between 0 and 1, learned through backpropagation, with a separate coefficient per hidden neuron.
- Exact endpoint attainment during learning is not required; fixed endpoints remain proposed controls.
- Activation computation cost matters. Use the linear/quadratic blend in place of the original continuous-power proposal. No general power is needed on activation values; actual runtime benefits must still be measured.
- CPU/CUDA execution and a scikit-learn style interface.
- A substantially smaller library than PSANN, with selective reuse of its useful work.
- ExpoActivation is implemented; later model, estimator, training, persistence, and benchmark work remains pending.

## Questions to settle

| ID | Decision | Proposed starting point | Tradeoff / alternative | Status |
| --- | --- | --- | --- | --- |
| D01 | Coefficient sharing | One blend coefficient per hidden neuron. | Confirmed for the primary model, carrying forward the per-neuron requirement. Per-layer sharing remains a proposed experimental control. | Primary behavior accepted |
| D02 | Bounds and initialization | `a` lies in `[0, 1]`; exact endpoint attainment is unnecessary. Propose `a = sigmoid(theta)` with `blend_init=0.5` and fixed endpoint controls. | The blend and its bounds are accepted. Mapping and initialization remain proposals; sigmoid runs on the small parameter vector before broadcasting. | Partially accepted |
| D03 | First supported workloads | Dense numeric tabular regression, including multiple outputs, plus binary and multiclass classification. | Build regression first. Sequence and image architectures would substantially enlarge the scope; callers can use the standalone activation in their own PyTorch models. | Proposed |
| D04 | Normalization | Optional feature standardization and optional hidden LayerNorm before the activation. Start with both enabled, then assess the hidden-normalization default experimentally. | LayerNorm can discard useful magnitude information, especially in narrow networks. Keep `normalization="none"` available. BatchNorm and RMSNorm are deferred. | Proposed |
| D05 | Convenience features | Mini-batches, validation, early stopping, history, separate blend learning rate, gradient clipping, and inference save/load. | Exact training resume, streaming, custom callbacks, and a deployment framework are outside the first release. | Proposed |
| D06 | Dependency boundary | Require PyTorch, NumPy, and scikit-learn. Use scikit-learn's real estimator machinery. | An optional sklearn extra reduces the base install but requires separate import/support paths. Prefer fewer maintenance paths initially. | Proposed |
| D07 | Execution scope | CPU and a single CUDA device in float32; CPU float64 for low-level mathematical tests. | AMP, compilation, distributed training, and other accelerators remain future work with separate evidence. | Proposed |
| D08 | Reuse and license | ExpoNet uses the MIT License. Adapt small verified PSANN components or tests only with required source attribution. | MIT keeps reuse and distribution straightforward and aligns with the referenced PSANN license. Do not copy entire subsystems to retain incidental features. | Accepted |
| D09 | Activation equation and cost | Replace the original power with `(1-a)*relu(x) + a*relu(x)*relu(x)`. | The blend is the sole supported activation family. Do not retain an exact-power mode. Keep an early timing check; no runtime speedup or overhead budget has been established. | Accepted |

The remaining main product questions are D02-D05. D06 and D07 retain proposed defaults pending confirmation. D08 and D09 are resolved; a measured performance budget can be set after initial timing.

## Experimental choices, not claims

- `blend_init=0.5` gives the sigmoid mapping its largest local derivative. It is a neutral starting point, not an established optimum. Compare against `0.1` and `0.9` later.
- Hidden LayerNorm is a candidate default, not a prerequisite for differentiability or a guarantee of stability.
- Start without residual connections or dropout. Add architecture features only after the activation comparison demonstrates a need.
- Improvements over ReLU, fixed blends, or PSANN are unknown. A correct implementation and an informative negative result still satisfy the initial research objective.

## Decision history

| Date | IDs | Outcome |
| --- | --- | --- |
| 2026-08-28 | D01-D08 | Initial proposals recorded; no implementation approval or default confirmation recorded. |
| 2026-08-28 | D01, D02, D09 | Per-neuron exponents and no requirement to reach exact endpoints confirmed. Computational cost raised as a design priority; sigmoid mapping, initialization, and any alternative equation remain unapproved. |
| 2026-08-28 | D01, D02, D09 | ReLU/squared-ReLU blend selected in place of the power equation. Per-neuron coefficients and `[0, 1]` blend bounds replace exponent semantics. Prior power-specific proposals are superseded; coefficient mapping and remaining defaults still need confirmation. |
| 2026-08-30 | D08 | Selected the MIT License for ExpoNet with `Copyright (c) 2026 ExpoNet Contributors`. This aligns with the referenced PSANN source license and supports straightforward reuse and distribution. |

When a decision changes, record the reason and affected roadmap tasks. Do not silently change the activation equation, parameter bounds, normalization order, or prediction shapes during implementation.
