import torch

from geodiff.sampling import farthest_point_sampling


def sample_T(geometry_dim: int, num_pts: int, device: str = 'cpu'):
    r"""Sample T values in [0, 1]^(d - 1) for d dimensional geometry. These are points that are
    mapped to our final shape.

    Args:
        geometry_dim: Dimension of the final output geometry.
        num_pts: Total points to sample.
        device: The device on which to create the candidate points.
    """

    if geometry_dim == 2:
        T = torch.linspace(0, 1, num_pts, device = device).reshape(-1, 1)
    elif geometry_dim == 3:
        T = farthest_point_sampling(num_pts = num_pts, device = device)

    return T