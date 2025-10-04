import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def _n_for_sin_peak_locations(x_peaks: torch.Tensor, eps: float = 1e-2) -> list[int]:
    r"""Convert sine-peak locations to Hicks-Henne exponents using :math:`n = ln(0.5)/ln(x_peak)`.

    Clamp values to avoid log related singularities at 0 and 1.

    Args:
        x_peaks: Basis function peak location. Shape :math:`(N,)`.
        eps: Optional clamp tolerance. Default: 1e-2.
    
    Returns:
        list[int]: Exponents `n_list` corresponding to each element of `x_peaks`.
    """
    # Clamp peak values on the left and right to avoid numerical errors in log computation
    x_peaks = torch.clamp(x_peaks, min = eps, max = 1 - eps)
    half = torch.tensor(0.5, dtype = x_peaks.dtype, device = x_peaks.device)
    n_vals = torch.log(half) / torch.log(x_peaks)

    return n_vals.tolist()


def _polyexp_basis(x: torch.Tensor, n_vals: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    r"""Polynomial exponential Hicks-Henne basis (Eq. 3 in [1]_).

    For each x value in `x`, n value in `n_vals` and given m, computes:
    .. math::
        y = \frac{\left(x \right)^{n} \left(1 - x \right)}{e^{mx}}
    
    Args:
        x: x-coordinates of points on the shape. Shape :math:`(N,)`.
        n_vals: Exponents n in the numerator polynomial. Shape :math:`(K,)`.
        m: Exponent of the exponential in the denominator. Scalar.
    
    Returns:
        torch.Tensor: Basis matrix. Shape (N, K). Rows = x-samples, Cols = n-values.
    
    References:
        [1] Hicks, Raymond M., and Garret N. Vanderplaats. Application of numerical optimization to
            the design of supercritical airfoils without drag-creep. No. 770440. SAE Technical
            Paper, 1977.
    """
    # Use torch pow function to compute all basis functions at once
    # N - count(x), K - count(n_vals), torch.pow(x, n_vals) returns (N, K)
    # Make x (N, 1), n_vals (1, K) and use broadcasting to produce an output (N, K)
    N = x.shape[0]
    n_vals = n_vals.view(1, -1)
    x = x.view(N, 1)
    num = torch.pow(x, n_vals) * (1.0 - x)
    den = torch.exp(m * x)
    polyexp_basis = num / den
    
    return polyexp_basis


def _sin_basis(x: torch.Tensor, n_vals: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    r"""Sinusoidal Hicks-Henne basis (Eq. 4 in [1]_).

    For each x value in `x`, n value in `n_vals` and given m, computes:
    .. math::
        y = \sin \left( \pi x^{n} \right)^{m}
    
    Args:
        x: x-coordinates of points on the shape. Shape :math:`(N,)`.
        n_vals: Exponents n of the input to the sine. Shape :math:`(K,)`.
        m: Exponent of the sine output. Scalar.
    
    Returns:
        torch.Tensor: Basis matrix. Shape (N, K). Rows = x-samples, Cols = n-values.
    
    References:
        [1] Hicks, Raymond M., and Garret N. Vanderplaats. Application of numerical optimization to
            the design of supercritical airfoils without drag-creep. No. 770440. SAE Technical
            Paper, 1977.
    """
    # Use torch pow function to compute all basis functions at once
    # N - count(x), K - count(n_vals), torch.pow(x, n_vals) returns (N, K)
    # Make x (N, 1), n_vals (1, K) and use broadcasting to produce an output (N, K)
    N = x.shape[0]
    n_vals = n_vals.view(1, -1)
    x = x.view(N, 1)
    z = torch.pow(x, n_vals)
    sin_basis = torch.pow(torch.sin(torch.pi * z), m)
    
    return sin_basis


class HicksHenne(nn.Module):
    r"""Hicks-Henne bump function representation. As described in [1]_.

    This module implements the Hicks-Henne bump functions as described in [1]_ and popularized by
    [2]_. It supports two families of basis functions: polynomial-exponential (Eq. 3 in [1]_) and
    sinusoidal (Eq. 4 in [1]_). These basis functions are applied separately to the upper and lower
    surfaces via learnable participation coefficients, allowing for flexible shape modifications
    during optimization.

    The basis functions are applied as linear additions to the baseline geometry as in the formulas
    below:
    .. math::
        y_{upper} = y_{upper}^{baseline} + \sum_i a_i f_i
        y_{lower} = y_{lower}^{baseline} + \sum_i b_i g_i
    
    The `a_i` and `b_i` above are the learnable parameters. There is a set for each basis family,
    that is, one set of `a_i`s for the polynomial-exponential basis functions and one for the
    sinusoidal basis functions. Similarly for `b_i`s.

    The basis functions are precomputed for efficiency and stored as buffers. The forward pass
    computes y-offsets by weighting the basis functions and adds them to the baseline coordinates.

    .. note::
        The implementation assumes that the x-coordinates lie in the range [0, 1].

    References:
        [1] Hicks, Raymond M., and Garret N. Vanderplaats. Application of numerical optimization to
            the design of supercritical airfoils without drag-creep. No. 770440. SAE Technical
            Paper, 1977.
        [2] Hicks, Raymond M., and Preston A. Henne. "Wing design by numerical optimization."
            Journal of aircraft 15.7 (1978): 407-412.
    """

    def __init__(
        self,
        X_upper_baseline: np.ndarray | torch.Tensor,
        X_lower_baseline: np.ndarray | torch.Tensor,
        polyexp_m: float,
        polyexp_n_list: list[int],
        sin_m: float,
        sin_n_list: list[int] | None = None,
        sin_n_count: int | None = None,
    ) -> None:
        r"""Initialize the Hicks-Henne representation.

        Args:
            X_upper_baseline: Upper surface baseline coordinates. Shape :math:`(N_upper, 2)`.
            X_lower_baseline: Lower surface baseline coordinates. Shape :math:`(N_lower, 2)`.
            polyexp_m: Scalar m for the polynomial-exponential basis family.
            polyexp_n_list: Exponent n list for the polynomial-exponential basis family.
            sin_m: Scalar m for the sinusoidal basis family.
            sin_n_list: Optional exponent n list for the sinusoidal basis family. Mutually exclusive
                with `sin_n_count`.
            sin_n_count: If provided, auto-generate this many sine bumps with even spaced peak
                locations in [0, 1] and convert to exponents n list.
        """
        super().__init__()

        if isinstance(X_upper_baseline, np.ndarray):
            X_upper_baseline = torch.from_numpy(X_upper_baseline)
        if isinstance(X_lower_baseline, np.ndarray):
            X_lower_baseline = torch.from_numpy(X_lower_baseline)

        # Check that x is between [0, 1] for the upper and lower surfaces
        if not torch.all((X_upper_baseline[:, 0] >= 0.0) & (X_upper_baseline[:, 0] <= 1.0)):
            raise ValueError('X_upper x-values must be in [0, 1].')
        if not torch.all((X_lower_baseline[:, 0] >= 0.0) & (X_lower_baseline[:, 0] <= 1.0)):
            raise ValueError('X_lower x-values must be in [0, 1].')
        
        
        # Save attributes in buffer so that they can be saved with state_dict
        self.register_buffer('X_upper_baseline', X_upper_baseline)
        self.register_buffer('X_lower_baseline', X_lower_baseline)
        # Convert exponents to tensors to store in register_buffer
        self.register_buffer('polyexp_m', torch.tensor(polyexp_m))
        self.register_buffer('polyexp_n_vals', torch.tensor(polyexp_n_list))
        self.register_buffer('sin_m', torch.tensor(sin_m))

        if sin_n_list is not None:
            self.register_buffer('sin_n_vals', torch.tensor(sin_n_list))
        else:
            if sin_n_count is None:
                raise ValueError('Provide either sin_n_list or sin_n_count.')
            x_peaks = torch.linspace(0.0, 1.0, sin_n_count)
            sin_n_list = _n_for_sin_peak_locations(x_peaks)
            self.register_buffer('sin_n_vals', torch.tensor(sin_n_list))
        

        # Learnable participation coefficients (K = count(n_vals) for each family)
        self.a_polyexp = nn.Parameter(torch.zeros(self.polyexp_n_vals.numel()))
        self.b_polyexp = nn.Parameter(torch.zeros(self.polyexp_n_vals.numel()))
        self.a_sin = nn.Parameter(torch.zeros(self.sin_n_vals.numel()))
        self.b_sin = nn.Parameter(torch.zeros(self.sin_n_vals.numel()))
        
        
        # Precompute the basis matrices to avoid repeated computation during forward()
        # This stores the basis values in a (N, K) matrix where
        # N - count(x), K - count(n_vals)
        self.register_buffer(
            'phi_upper_polyexp',
            _polyexp_basis(self.X_upper_baseline[:, 0], self.polyexp_n_vals, self.polyexp_m),
            persistent = False
        )
        self.register_buffer(
            'phi_lower_polyexp',
            _polyexp_basis(self.X_lower_baseline[:, 0], self.polyexp_n_vals, self.polyexp_m),
            persistent = False
        )
        self.register_buffer(
            'phi_upper_sin',
            _sin_basis(self.X_upper_baseline[:, 0], self.sin_n_vals, self.sin_m),
            persistent = False
        )
        self.register_buffer(
            'phi_lower_sin',
            _sin_basis(self.X_lower_baseline[:, 0], self.sin_n_vals, self.sin_m),
            persistent = False
        )


    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute Hicks-Henne y-offsets for the upper and lower surfaces.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Upper and lower surface coordinates. `X_upper` of
                shape :math:`(N_upper, 2)` and `X_lower` of shape :math:`(N_lower, 2)`.
        """
        # Apply the bump function offsets to the upper and lower surfaces
        dy_upper = self.phi_upper_polyexp @ self.a_polyexp + self.phi_upper_sin @ self.a_sin
        dy_lower = self.phi_lower_polyexp @ self.b_polyexp + self.phi_lower_sin @ self.b_sin

        X_upper = torch.stack([self.X_upper_baseline[:, 0], self.X_upper_baseline[:, 1] + dy_upper],
                              dim = 1)
        X_lower = torch.stack([self.X_lower_baseline[:, 0], self.X_lower_baseline[:, 1] + dy_lower],
                              dim = 1)

        return X_upper, X_lower


    def visualize(self, ax = None):
        r"""Plot baseline and Hicks-Henne geometry.

        Args:
            ax: Optional Matplotlib Axes to draw on. If `None`, a new figure/axes is created.
        
        Returns:
            tuple[fig, ax]: The figure and axes used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Move to CPU for matplotlib
        X_upper_baseline = self.X_upper_baseline.detach().cpu()
        X_lower_baseline = self.X_lower_baseline.detach().cpu()
        X_upper, X_lower = self.forward()
        X_upper, X_lower = X_upper.detach().cpu(), X_lower.detach().cpu()

        # Plot the baseline shape
        ax.plot(X_upper_baseline[:, 0], X_upper_baseline[:, 1], linestyle = '-', linewidth = 2,
                color = 'black', alpha = 0.7, label = 'upper (baseline)')
        ax.plot(X_lower_baseline[:, 0], X_lower_baseline[:, 1], linestyle = '--', linewidth = 2,
                color = 'black', alpha = 0.7, label = 'lower (baseline)')

        # Plot the shape
        ax.plot(X_upper[:, 0], X_upper[:, 1], linestyle = '-', linewidth = 2,
                color = 'orange', alpha = 0.7, label = 'upper')
        ax.plot(X_lower[:, 0], X_lower[:, 1], linestyle = '--', linewidth = 2,
                color = 'orange', alpha = 0.7, label = 'lower')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title('Hicks-Henne Bump Function Parameterization')
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.12), ncol = 2)

        return fig, ax