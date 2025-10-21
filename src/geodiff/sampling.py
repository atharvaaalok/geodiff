import torch


def farthest_point_sampling(
    num_pts: int,
    oversampling_factor: int = 16,
    seed: int = 42,
    device: str = 'cpu'
) -> torch.Tensor:
    r"""Greedy Farthest Point Sampling (FPS). Picks exactly `num_pts` samples, blue-noise-like.

    Args:
        num_pts: Total number of points to sample in the domain.
        oversampling_factor: Total points to randomly sample out of which the `num_pts` farthest
            subset will be chosen.
        seed: Random seed to use for generating the points.
        device: The device on which to create the candidate points.
    """

    rng = torch.Generator(device = device).manual_seed(seed)
    # Oversample candidate points randomly to select FPS subset from
    M = num_pts * oversampling_factor
    candidates = torch.rand((M, 2), generator = rng, device = device)

    selected_idx = torch.empty(num_pts, dtype = torch.long, device = device)
    # Start by selecting first candidate as first sample
    selected_idx[0] = 0
    last_selected_pt = candidates[selected_idx[0]]

    # Compute squared distance to nearest selected point so far
    d2 = torch.full((M,), float('inf'), device = device)

    for i in range(1, num_pts):
        # Update nearest-distance-to-set with the last chosen point
        d2 = torch.minimum(d2, torch.sum((candidates - last_selected_pt) ** 2, dim = 1))
        idx = torch.argmax(d2)
        selected_idx[i] = idx
        last_selected_pt = candidates[idx]

    return candidates[selected_idx]