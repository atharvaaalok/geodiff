import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def _n_for_sin_peak_locations(x_peaks: torch.Tensor, eps: float = 1e-2) -> list[int]:
    # Clamp peak values on the left and right to avoid numerical errors in log computation
    x_peaks = torch.clamp(x_peaks, min = eps, max = 1 - eps)
    half = torch.tensor(0.5, dtype = x_peaks.dtype, device = x_peaks.device)
    n_vals = torch.log(half) / torch.log(x_peaks)

    return n_vals.tolist()


def _polyexp_basis(x: torch.Tensor, n_vals: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
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
        # Apply the bump function offsets to the upper and lower surfaces
        dy_upper = self.phi_upper_polyexp @ self.a_polyexp + self.phi_upper_sin @ self.a_sin
        dy_lower = self.phi_lower_polyexp @ self.b_polyexp + self.phi_lower_sin @ self.b_sin

        X_upper = torch.stack([self.X_upper_baseline[:, 0], self.X_upper_baseline[:, 1] + dy_upper],
                              dim = 1)
        X_lower = torch.stack([self.X_lower_baseline[:, 0], self.X_lower_baseline[:, 1] + dy_lower],
                              dim = 1)

        return X_upper, X_lower


    def visualize(self, ax = None):
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
        ax.plot(X_upper_baseline[:, 0], X_upper_baseline[:, 1], linestyle = '--', linewidth = 2,
                color = 'black', alpha = 0.7, label = 'upper (baseline)')
        print(X_upper_baseline[:, 0])
        ax.plot(X_lower_baseline[:, 0], X_lower_baseline[:, 1], linestyle = '--', linewidth = 2,
                color = 'black', alpha = 0.7, label = 'lower (baseline)')

        # Plot the shape
        ax.plot(X_upper[:, 0], X_upper[:, 1], linestyle = '-', linewidth = 2,
                color = 'orange', alpha = 0.7, label = 'upper')
        ax.plot(X_lower[:, 0], X_lower[:, 1], linestyle = '-', linewidth = 2,
                color = 'orange', alpha = 0.7, label = 'lower')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title('Hicks-Henne Bump Function Parameterization')
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), ncol = 2)

        return fig, ax