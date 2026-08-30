"""Fit ExpoRegressor on a small synthetic numeric regression problem."""

import numpy as np
import torch

from exponet import ExpoRegressor


def main() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(80, 2)).astype(np.float32)
    targets = (1.5 * features[:, 0] - 0.75 * features[:, 1] + 0.2).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ExpoRegressor(
        hidden_dims=(8,),
        normalization="none",
        trainable_blend=False,
        blend_init=0.0,
        epochs=80,
        lr=0.03,
        device=device,
        random_state=0,
    ).fit(features, targets)
    print(f"device={model.device_} r2={model.score(features, targets):.3f}")


if __name__ == "__main__":
    main()
