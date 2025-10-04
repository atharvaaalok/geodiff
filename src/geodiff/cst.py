import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def _bernstein_basis(x: torch.Tensor, n: int) -> torch.Tensor:
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
    return torch.pow(x, n1) * torch.pow(1 - x, n2)


class CST(nn.Module):

    def __init__(
        self,
        n1: float = 0.5,
        n2: float = 1.0,
        upper_basis_count: int = 9,
        lower_basis_count: int = 9,
        upper_te_thickness: float = 0.0,
        lower_te_thickness: float = 0.0,
    ) -> None:
        
        super().__init__()

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
        y_lower = C * S_lower + self.lower_te_thickness

        # Return the (x, y) coordinate pairs of points on the shape
        X_upper = torch.stack([x, y_upper], dim = 1)
        X_lower = torch.stack([x, y_lower], dim = 1)

        return X_upper, X_lower
    

    def visualize(self, x: torch.Tensor = None, num_pts: int = 100, ax = None):
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