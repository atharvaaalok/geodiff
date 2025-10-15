import math

import torch


def closed_transform_2d(t: torch.Tensor) -> torch.Tensor:
    r"""Map :math:`t` in :math:`[0, 1]` to points :math:`(x, y)` on the unit circle.

    Map :math:`t \in [0, 1]` to:
    .. math::
        \theta &= 2 \pi t \\
        x &= \cos \theta \\
        y &= \sin \theta

    Args:
        t: Tensor of values in [0, 1]. Shape :math:`(N, 1)`.

    Returns:
        torch.Tensor: Matrix of coordinates (x, y) on the unit circle. Shape :math:`(N, 2)`.
    """
    pi = t.new_tensor(math.pi)
    theta = 2 * pi * t
    x = torch.cos(theta)
    y = torch.sin(theta)

    return torch.cat([x, y], dim = 1)


def closed_transform_3d(ts: torch.Tensor) -> torch.Tensor:
    r"""Map :math:`(t, s)` in :math:`[0, 1]^2` to points :math:`(x, y, z)` on the unit sphere.

    Map :math:`(t, s) \in [0, 1]^2` to:
    .. math::
        \theta &= 2 \pi t \qquad \phi = \pi s
        x &= \sin \phi \cos \theta \\
        y &= \sin \phi \sin \theta \\
        z &= \cos \phi

    Args:
        ts: Tensor of (t, s) values in [0, 1]. Shape :math:`(N, 2)`.

    Returns:
        torch.Tensor: Matrix of coordinates (x, y, z) on the unit sphere. Shape :math:`(N, 3)`.
    """
    t = ts[:, 0:1]
    s = ts[:, 1:2]

    pi = t.new_tensor(math.pi)
    theta = 2 * pi * t
    phi = pi * s

    sin_phi = torch.sin(phi)
    x = sin_phi * torch.cos(theta)
    y = sin_phi * torch.sin(theta)
    z = torch.cos(phi)

    return torch.cat([x, y, z], dim = 1)