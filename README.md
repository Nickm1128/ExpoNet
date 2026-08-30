# ExpoNet

ExpoNet is a planned, compact PyTorch library for networks with a trainable blend of ReLU and squared ReLU:

```text
u = max(x, 0)
f(x; a) = (1 - a) * u + a * u * u
0 <= a <= 1, with a separate learned coefficient per hidden neuron
```

The goal is to investigate whether learning activation curvature improves useful models over a comparable ReLU network, while retaining straightforward CPU/CUDA execution and a scikit-learn estimator interface.

The blend replaces the original continuous-power proposal. Its activation-value arithmetic uses multiplication and addition rather than general powers. The coefficient constraint and other defaults remain under review; actual runtime benefits require measurement.

**Status: ExpoActivation is implemented and passes the current CPU validation suite.** It supports per-neuron and per-layer coefficients, fixed and trainable modes, axis-aware broadcasting, strict state-dict loading, gradient checks, and isolated coefficient recovery. The MLP, estimators, training loop, persistence, benchmarks, and other later phases are not implemented. CUDA validation is prepared but deferred because compatible hardware is unavailable. ExpoNet is available under the [MIT License](LICENSE); activation-overhead measurement and CUDA validation remain pending, so P1 is not complete.

## Planning documents

| Document | Purpose |
| --- | --- |
| [Open decisions](docs/DECISIONS.md) | Confirm scope and defaults before coding. |
| [Design](docs/DESIGN.md) | Mathematics, numerical behavior, normalization, and architecture. |
| [API contract](docs/API_CONTRACT.md) | Proposed interfaces and exact training/prediction semantics. |
| [Roadmap](docs/ROADMAP.md) | Ordered tasks, dependencies, acceptance criteria, and progress. |
| [Validation](docs/VALIDATION.md) | Correctness checks and controlled experiments. |
| [PSANN reuse assessment](docs/PSANN_REUSE.md) | What to adapt, what to leave behind, and source provenance. |

Start with the open decisions, then follow the roadmap one task at a time. Planning documents describe intended behavior except where the roadmap records completed activation work. No performance or distribution-readiness claim is made.
