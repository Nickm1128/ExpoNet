"""Train ExpoMLP directly with a caller-owned PyTorch optimizer."""

import torch

from exponet import ExpoMLP


def main() -> None:
    torch.manual_seed(0)
    model = ExpoMLP(2, 1, hidden_dims=(4,), normalization="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    features = torch.tensor([[0.5, 1.0], [1.0, 2.0], [1.5, 0.5]])
    targets = torch.tensor([[0.0], [1.0], [-1.0]])
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(features), targets)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        prediction = model(features[:1])
    print(f"prediction_shape={tuple(prediction.shape)}")


if __name__ == "__main__":
    main()
