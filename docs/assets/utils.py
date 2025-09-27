import matplotlib.pyplot as plt
import torch


def circle(num_pts: int) -> torch.Tensor:
    theta = torch.linspace(0, 2 * torch.pi, num_pts)
    X = torch.stack([torch.cos(theta), torch.sin(theta)], dim = 1)
    # Move last point to front, this avoids the jump from x: 0 to 1 when splitting upper and lower
    X = torch.vstack([X[-1], X[:-1]])
    return X


def square(num_pts: int) -> torch.Tensor:
    # Generate points on the unit circle first and then map them to a square
    theta = torch.linspace(0, 2 * torch.pi, num_pts)

    x, y = torch.cos(theta), torch.sin(theta)
    s = torch.maximum(torch.abs(x), torch.abs(y))
    x_sq, y_sq = x/s, y/s

    X = torch.stack([x_sq, y_sq], dim = 1)
    # Move last point to front, this avoids the jump from x: 0 to 1 when splitting upper and lower
    X = torch.vstack([X[-1], X[:-1]])
    return X


def normalize_0_to_1(X: torch.Tensor):
    # Normalize x to the range [0, 1] and appropriately scale y
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    X_norm = X.clone()
    X_norm[:, 0] = (X_norm[:, 0] - x_min) / (x_max - x_min)
    X_norm[:, 1] = X_norm[:, 1] / (x_max - x_min)
    return X_norm