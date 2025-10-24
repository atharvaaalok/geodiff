import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

from geodiff.transforms import closed_transform_2d, closed_transform_3d
from geodiff.utils import sample_T


class NeuralODE(nn.Module):
    r"""Neural Ordinary Differential Equation (NeuralODE) architecture as described in [1]_.

    This module generates geometry by evolving a unit circle according a learnable ordinary
    differential equation. The NeuralODE forward and backward passes are handled using the
    `torchdiffeq` library [2]_.

    .. math:
        TODO

    References:
        [1]: Chen, Ricky TQ, et al. "Neural ordinary differential equations." Advances in neural
            information processing systems 31 (2018).
        [2]: Chen, Ricky TQ. "torchdiffeq, 2018." URL https://github.com/rtqichen/torchdiffeq 14.
    """

    def __init__(
        self,
        geometry_dim: int,
        ode_net: nn.Module,
        preaux_net: nn.Module,
        solver: str = 'dopri5',
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ) -> None:
        r"""Initialize the NeuralODE object.

        Args:
            geometry_dim: Dimension of the output geometry, e.g. 2d, 3d etc.
            ode_net: A torch network for the function defining the neural ordinary differential
                equation.
        """
        super().__init__()

        # Save ode solver properties
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Save attributes in buffer so that they can be saved with state_dict
        self.register_buffer('geometry_dim', torch.tensor(geometry_dim, dtype = torch.int64))


        # Use a Pre-Aux Net to create baseline closed manifold
        self.preaux_net = copy.deepcopy(preaux_net)


        # Create an object of ODEFunc class to use with NeuralODE forward pass using the provided
        # ODE network definition
        self.ode_f = ODEFunc(geometry_dim = geometry_dim, ode_net = ode_net)


    def forward(
        self,
        T: torch.Tensor = None,
        num_pts: int = 1000,
        code: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Compute the coordinates of the shape represented by the NeuralODE object.

        Args:
            T: Options samples in the input domain which are mapped to points on the geometry.
                Mutually exclusive with `num_pts`.
            num_pts: Optional number of points :math:`N` to generate on the surface. Mututally
                exclusive with `T`.

        Returns:
            torch.Tensor: Matrix of coordinates of points on the geometry. Shape :math:`(N, d)`,
                where :math:`d` is the dimension of the geometry.
        """
        device = next(self.parameters()).device
        if T is None:
            T = sample_T(geometry_dim = self.geometry_dim, num_pts = num_pts, device = device)
        
        if self.ode_f.ode_net.input_dim != (self.geometry_dim + 1):
            # If a single code is provided create copies of it to match the number of T values
            code = code.to(device)
            if T.shape[0] != code.shape[0]:
                code = code.expand(T.shape[0], -1)
            # Otherwise we assume that for each T the right latent vector has been put in code

        # First create a closed transform using the PreAux net
        closed_manifold = self.preaux_net(T)

        # Use closed manifold as initial condition - (N, d)
        y0 = closed_manifold

        # Define time integration limits
        time = torch.tensor([0.0, 1.0], device = device)

        # Concatenate initial state and code if code is required
        if code is not None:
            y0_with_code = torch.cat([y0, code], dim = -1)
        else:
            y0_with_code = y0

        Y = odeint(self.ode_f, y0_with_code, time, method = self.solver,
                   rtol = self.rtol, atol = self.atol,
                   options = {'dtype': torch.float32}).to(device)
        # Get the final shape at time t = 1
        X = Y[-1, :, :self.geometry_dim]

        return X


    def visualize(
        self,
        T: torch.Tensor = None,
        num_pts: int = 1000,
        code: torch.Tensor = None,
        ax = None,
    ):
        r"""Plot geometry represented by the NeuralODE object.

        Args:
            T: Options samples in the input domain which are mapped to points on the geometry.
                Mutually exclusive with `num_pts`.
            num_pts: Optional number of points :math:`N` to generate on the surface. Mututally
                exclusive with `T`.
            ax: Optional Matplotlib Axes to draw on. If `None`, a new figure/axes is created.

        Returns:
            tuple[fig, ax]: The figure and axes used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Move to CPU for matplotlib
        X = self.forward(T = T, num_pts = num_pts)
        X = X.detach().cpu()

        # Plot the shape
        ax.plot(X[:, 0], X[:, 1], linestyle = '-', linewidth = 2, color = 'orange', alpha = 0.7,
                label = 'curve')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title('NeuralODE parameterization')
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.12), ncol = 2)

        return fig, ax


class ODEFunc(nn.Module):
    r"""Right-hand side of the ODE: dX/dt = f_θ(X, t).
    """

    def __init__(self, geometry_dim: int, ode_net: nn.Module) -> None:
        r"""Initialize the ODEFunc object.
        
        This forms a thin wrapper around the user-provided `ode_net` to make the forward pass
        compatible with `odeint` function from `torchdiffeq`.

        Args:
            geometry_dim: Dimension of the output geometry, e.g. 2d, 3d etc.
            ode_net: A torch network for the function defining the neural ordinary differential
                equation.
        """
        super().__init__()

        self.geometry_dim = geometry_dim
        self.ode_net = copy.deepcopy(ode_net)


    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`dy/dt = f_\theta(y, t)`, where :math:`f_\theta` is the `ode_net`
        defining the ODE evolution.

        Args:
            t: Scalar time at which to compute the time rate of change of state.
            y: State tensor. Shape :math:`(..., d)`, where :math:`d` is the geometry dimension.
        """
        # Extract the state and the code from their concatenation
        geometry_dim = self.geometry_dim
        y, code = y[..., :geometry_dim], y[..., geometry_dim:]

        # Broadcast t to appropriate shape to concatenate with y
        t = torch.broadcast_to(t.to(y), y[..., :1].shape)

        # Concatenate y and t to feed as input to our ode function
        state_with_time = torch.cat([y, t, code], dim = -1)

        # Compute the rate of change of y using the neural network defining the ODE function
        dy = self.ode_net(state_with_time)

        # Set rate of codes to 0 to preserve their values
        dcode = torch.zeros_like(code)

        return torch.cat([dy, dcode], dim = -1)