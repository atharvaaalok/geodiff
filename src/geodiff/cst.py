import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def _bernstein_basis(x: torch.Tensor, n: int) -> torch.Tensor:
    r"""Compute the Bernstein basis functions of degree :math:`n`.

    For each x value in `x`, k value in :math:`k = 0, \dots, n`, the Bernstein polynomial is:
    .. math::
        B_{k, n}(x) \;=\; \binom{n}{k}\, x^{k}\, \left(1 - x \right)^{n-k}.

    Args:
        x: x-coordinates of points to evaluate basis functions at. Shape :math:`(N,)`.
        n: Non-negative integer degree of the Bernstein polynomials. Produces :math:`n{+}1` columns.

    Returns:
        torch.Tensor: Basis matrix. Shape (N, K). Rows = x-samples, Cols = k-values.
    """
    dtype, device = x.dtype, x.device

    # Compute all binomial coefficients (n choose k) and exponents
    nCk = [math.comb(n, k) for k in range(n + 1)]
    nCk = torch.tensor(nCk, dtype = dtype, device = device)
    k = torch.arange(n + 1, dtype = dtype, device = device)

    # Reshape tensors to compute all basis functions at once
    # N - count(x), n + 1 - count(k), basis matrix returned (N, n + 1)
    # Make x (N, 1), k (1, n + 1) and use broadcasting to produce an output (N, n + 1)
    N = x.shape[0]
    x = x.view(N, 1)
    k = k.view(1, -1)
    nCk = nCk.view(1, -1)

    # Compute the basis
    B_kn = nCk * torch.pow(x, k) * torch.pow(1.0 - x, n - k)

    return B_kn


def _class_function(x: torch.Tensor, n1: float, n2: float) -> torch.Tensor:
    r"""Evaluate the CST *class* function :math:`C(x) = x^{n_1}(1-x)^{n_2}`.

    Args:
        x: x-coordinates of points to evaluate class function at. Shape :math:`(N,)`.
        n1: Class function exponent :math:`n_1`.
        n2: Class function exponent :math:`n_2`.

    Returns:
        torch.Tensor: :math:`C(x)` with the same shape and dtype as `x`.
    """
    return torch.pow(x, n1) * torch.pow(1 - x, n2)


class CST(nn.Module):
    r"""Class-Shape Transformation (CST) representation as described in [1]_.

    This module implements the CST representation as described in [1]_. The upper and lower surfaces
    of the shape are defined by:
    .. math::
        y_u(x) &= C(x)\,S_u(x) + \tau_u, \\
        y_l(x) &= C(x)\,S_l(x) + \tau_l,

    with shape functions
    .. math::
        S_u(x) &= \sum_{i=0}^{K_u-1} A_{u,i}\,S_i(x), \qquad
        S_l(x) = \sum_{i=0}^{K_l-1} A_{l,i}\,S_i(x),

    where the element shape functions :math:`S_i(x)` are the Bernstein polynomials of
    degree :math:`n = K-1`,
    .. math::
        S_i(x) \;=\; \binom{n}{i}\,x^{\,i}\,(1-x)^{\,n-i}, \qquad i = 0, \ldots, n.

    The class function is
    .. math::
        C(x) \;=\; x^{\,n_1}\,\bigl(1-x\bigr)^{\,n_2}.

    The scalars :math:`\tau_u` and :math:`\tau_l` are trailing-edge offsets that
    govern the trailing-edge thickness; specifically
    .. math::
        t_{\mathrm{TE}} \;=\; \tau_u - \tau_l .
    
    By default, all coefficients :math:`A_{u,i}` and :math:`A_{l,i}` are initialized to 1, so that
    :math:`\sum_i S_i(x) = 1` (partition of unity) and the initial surfaces reduce to
    :math:`y_u(x) = C(x) + \tau_u` and :math:`y_l(x) = C(x) + \tau_l`.

    References:
        [1]: Kulfan, Brenda, and John Bussoletti. "" Fundamental" parameteric geometry
            representations for aircraft component shapes." 11th AIAA/ISSMO multidisciplinary
            analysis and optimization conference. 2006.
    """

    def __init__(
        self,
        n1: float = 0.5,
        n2: float = 1.0,
        upper_basis_count: int = 9,
        lower_basis_count: int = 9,
        upper_te_thickness: float = 0.0,
        lower_te_thickness: float = 0.0,
    ) -> None:
        r"""Initialize the Hicks-Henne representation.

        Args:
            n1: Optional Class-function exponent :math:`n_1` in :math:`C(x) = x^{n_1}(1-x)^{n_2}`.
            n2: Optional Class-function exponent :math:`n_2` in :math:`C(x) = x^{n_1}(1-x)^{n_2}`.
            upper_basis_count: Optional Number of Bernstein basis functions for the upper surface.
                Internally uses Bernstein polynomial of degree ``n = upper_basis_count - 1``.
            lower_basis_count: Optional Number of Bernstein basis functions for the lower surface.
                Internally uses Bernstein polynomial of degree ``n = lower_basis_count - 1``.
            upper_te_thickness: Optional Trailing-edge offset added to the upper surface.
            lower_te_thickness: Optional Trailing-edge offset added to the lower surface.
        
        Raises:
            ValueError: If either of the upper or lower trailing edge thickness is negative.
        """
        super().__init__()

        if (upper_te_thickness < 0.0) or (lower_te_thickness < 0.0):
            raise ValueError('The upper and lower trailing edge thickness should be non-negative.')

        # Save attributes in buffer so that they can be saved with state_dict
        self.register_buffer('n1', torch.tensor(n1))
        self.register_buffer('n2', torch.tensor(n2))
        self.register_buffer('upper_basis_count', torch.tensor(upper_basis_count,
                                                               dtype = torch.int64))
        self.register_buffer('lower_basis_count', torch.tensor(lower_basis_count,
                                                               dtype = torch.int64))
        self.register_buffer('upper_te_thickness', torch.tensor(upper_te_thickness))
        self.register_buffer('lower_te_thickness', torch.tensor(lower_te_thickness))


        # Learnable coefficients for the Shape functions
        self.A_upper = nn.Parameter(torch.ones(self.upper_basis_count.item()))
        self.A_lower = nn.Parameter(-torch.ones(self.lower_basis_count.item()))


    def forward(
        self,
        x: torch.Tensor = None,
        num_pts: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the CST y-coordinates for upper and lower surfaces.

        Args:
            x: Optional abscissae. Shape :math:`(N,)`, values in :math:`[0, 1]`. Mutually exclusive
                with `num_pts`.
            num_pts: Optional number of points :math:`N` to generate on each surface. Mutually
                exclusive with `x`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Upper and lower surface coordinates of shape
                :math:`(N, 2)`.
        
        Raises:
            ValueError: If any :math:`x \notin [0, 1]`.
        """
        
        # If x is provided check that it is in [0, 1] if not provided create equally spaced x
        if x is not None:
            if not torch.all((x >= 0.0) & (x <= 1.0)):
                raise ValueError('x-values must be in [0, 1].')
        else:
            dtype, device = self.n1.dtype, self.n1.device
            x = torch.linspace(0.0, 1.0, num_pts, dtype = dtype, device = device)
        
        # Get basis counts
        upper_basis_count = int(self.upper_basis_count.item())
        lower_basis_count = int(self.lower_basis_count.item())

        # Compute the class function and the Bernstein basis matrix
        C = _class_function(x, self.n1.item(), self.n2.item())
        B_upper = _bernstein_basis(x, upper_basis_count - 1)
        B_lower = _bernstein_basis(x, lower_basis_count - 1)

        # Compute the shape functions
        S_upper = B_upper @ self.A_upper
        S_lower = B_lower @ self.A_lower

        # Compute the y coordinates of points on the shape
        y_upper = C * S_upper + self.upper_te_thickness
        y_lower = C * S_lower - self.lower_te_thickness

        # Return the (x, y) coordinate pairs of points on the shape
        X_upper = torch.stack([x, y_upper], dim = 1)
        X_lower = torch.stack([x, y_lower], dim = 1)

        return X_upper, X_lower
    

    def visualize(self, x: torch.Tensor = None, num_pts: int = 100, ax = None):
        r"""Plot baseline and CST geometry.

        Args:
            x: Optional abscissae. Shape :math:`(N,)`, values in :math:`[0, 1]`. Mutually exclusive
                with `num_pts`.
            num_pts: Optional number of points :math:`N` to generate on each surface. Mutually
                exclusive with `x`.
            ax: Optional Matplotlib Axes to draw on. If `None`, a new figure/axes is created.
        
        Returns:
            tuple[fig, ax]: The figure and axes used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        # Move to CPU for matplotlib
        X_upper, X_lower = self.forward(x = x, num_pts = num_pts)
        X_upper, X_lower = X_upper.detach().cpu(), X_lower.detach().cpu()

        # Plot the shape
        ax.plot(X_upper[:, 0], X_upper[:, 1], linestyle = '-', linewidth = 2,
                color = 'orange', alpha = 0.7, label = 'upper')
        ax.plot(X_lower[:, 0], X_lower[:, 1], linestyle = '--', linewidth = 2,
                color = 'orange', alpha = 0.7, label = 'lower')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title('CST Parameterization')
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.12), ncol = 2)

        return fig, ax