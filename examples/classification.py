"""Fit ExpoClassifier on a small synthetic multiclass numeric dataset."""

import numpy as np

from exponet import ExpoClassifier


def main() -> None:
    generator = np.random.default_rng(1)
    centers = np.array([[-2.0, -2.0], [2.0, -1.0], [0.0, 2.0]], dtype=np.float32)
    features = np.concatenate(
        [
            center + 0.25 * generator.standard_normal((20, 2)).astype(np.float32)
            for center in centers
        ]
    )
    labels = np.repeat(np.array(["alpha", "beta", "gamma"]), 20)
    model = ExpoClassifier(
        hidden_dims=(8,),
        normalization="none",
        epochs=80,
        lr=0.03,
        device="auto",
        random_state=1,
    ).fit(features, labels)
    print(f"device={model.device_} accuracy={model.score(features, labels):.3f}")
    print(f"classes={model.classes_.tolist()}")
    print(f"probabilities={model.predict_proba(features[:1]).round(3).tolist()}")


if __name__ == "__main__":
    main()
