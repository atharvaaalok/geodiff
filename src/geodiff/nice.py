import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class NICE(nn.Module):
    r"""Nonlinear Independent Components Estimation (NICE) architecture as described in [1]_
    pre-composed with a Pre-Auxilliary network.

    This module generates a closed, non-self-intersecting base geometry via a PreAuxNet and then
    applies a stack of NICE layers.

    .. math:
        TODO

    References:
        [1]: Dinh, Laurent, David Krueger, and Yoshua Bengio. "Nice: Non-linear independent
            components estimation." arXiv preprint arXiv:1410.8516 (2014).
    """

    def __init__(
        self,
        geometry_dim: int,
        layer_count: int,
        preaux_net: nn.Module,
        coupling_net: nn.Module,
        volume_preserving: bool = False,
        use_batchnormalization: bool = False,
        use_residual_connection: bool = True,
    ) -> None:
        r"""Initialize the NICE object.

        Args:
            geometry_dim: Dimension of the output geometry, e.g. 2d, 3d etc.
            layer_count: Number of hidden layers in the NICE net.
            preaux_net: A torch network that is used as the Pre-Aux net.
            coupling_net: A torch network that is used as the coupling function.
            volume_preserving: If `False`, applies a learnable positive scaling at the output. If
                `True`, no scaling is applied and the transformation is volume preserving.
            use_batchnormalization: If `True`, applies batch normalization after each coupling
                layer.
            use_residual_connection: If `True`, applies a residual connection after each coupling
                layer.
        """
        super().__init__()

        # Save attributes in buffer so that they can be saved with state_dict
        self.register_buffer('geometry_dim', torch.tensor(geometry_dim, dtype = torch.int64))
        self.register_buffer('layer_count', torch.tensor(layer_count, dtype = torch.int64))
        self.register_buffer('volume_preserving', torch.tensor(volume_preserving,
                                                               dtype = torch.bool))
        self.register_buffer('use_batchnormalization', torch.tensor(use_batchnormalization,
                                                               dtype = torch.bool))
        self.register_buffer('use_residual_connection', torch.tensor(use_residual_connection,
                                                               dtype = torch.bool))


        # Use a Pre-Aux Net to create baseline closed manifold
        self.preaux_net = copy.deepcopy(preaux_net)


        # Create a base mask to define the first partition
        d_partition = (geometry_dim // 2) if (geometry_dim % 2 == 0) else (geometry_dim + 1) // 2
        mask1 = torch.tensor([True] * d_partition + [False] * (geometry_dim - d_partition),
                                 dtype = torch.bool)
        mask2 = torch.tensor([False] * (geometry_dim - d_partition) + [True] * d_partition,
                                 dtype = torch.bool)
        self.register_buffer('mask1', mask1)
        self.register_buffer('mask2', mask2)


        # Create coupling nets for each NICE layer
        self.coupling_nets = nn.ModuleList()
        for i in range(layer_count):
            self.coupling_nets.append(copy.deepcopy(coupling_net))


        # Use batchnormalization with each coupling layer
        self.normalization_layers = nn.ModuleList()
        for i in range(layer_count):
            if use_batchnormalization:
                self.normalization_layers.append(nn.BatchNorm1d(geometry_dim))
            else:
                self.normalization_layers.append(nn.Identity())


        # Use positive diagonal scaling
        if not volume_preserving:
            self.log_scale = nn.Parameter(torch.zeros(geometry_dim))


    def forward(
        self,
        T: torch.Tensor = None,
        num_pts: int = 1000,
        code: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Compute the coordinates of the shape represented by the NICE object.

        Args:
            T: Options samples in the input domain which are mapped to points on the geometry.
                Mutually exclusive with `num_pts`.
            num_pts: Optional number of points :math:`N` to generate on the surface. Mututally
                exclusive with `T`.

        Returns:
            torch.Tensor: Matrix of coordinates of points on the geometry. Shape :math:`(N, d)`,
                where :math:`d` is the dimension of the geometry.
        """
        if T is None:
            if self.geometry_dim == 2:
                T = torch.linspace(0, 1, num_pts).reshape(-1, 1)
            elif self.geometry_dim == 3:
                t, s = torch.meshgrid(torch.linspace(0, 1, num_pts), torch.linspace(0, 1, num_pts),
                                      indexing = 'ij')
                T = torch.stack([t, s], dim = -1).reshape(-1, 2)
        # Put T on the same device as the model
        device = next(self.parameters()).device
        T.to(device = device)

        if self.preaux_net.latent_dim != 0:
            # If a single code is provided create copies of it to match the number of T values
            code = code.to(device)
            if T.shape[0] != code.shape[0]:
                code = code.expand(T.shape[0], -1)
            # Otherwise we assume that for each T the right latent vector has been put in code

        # First create a closed transform using the PreAux net
        X = self.preaux_net(T, code)

        # Apply the NICE coupling layers
        mask_switch = True
        for coupling_net, normalization_layer in zip(self.coupling_nets, self.normalization_layers):
            # Pick a mask for the partition
            mask = self.mask1 if mask_switch else self.mask2

            if self.use_residual_connection:
                residual = X

            # Create the partition
            x1 = X[:, mask]
            x2 = X[:, ~mask]

            # Compute the output of the additive coupling layer
            y1 = x1
            y2 = x2 + coupling_net(x1)

            # Assign the y elements to the right place
            Y = torch.empty_like(X)
            Y[:, mask] = y1
            Y[:, ~mask] = y2

            # Assign Y to X to be used in the next layer as input
            X = Y

            # Apply batchnormalization
            X = normalization_layer(X)

            if self.use_residual_connection:
                # Apply residual connection
                X = (X + residual) / 2.0

            # Invert the mask for the next layer
            mask_switch = not mask_switch

        # Apply the final scaling if non-volume preserving
        if not self.volume_preserving:
            X = X * torch.exp(self.log_scale)

        return X


    def visualize(
        self,
        T: torch.Tensor = None,
        num_pts: int = 1000,
        code: torch.Tensor = None,
        ax = None,
    ):
        r"""Plot geometry represented by the NICE object.

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
        X = self.forward(T = T, num_pts = num_pts, code = code)
        X = X.detach().cpu()

        # Plot the shape
        ax.plot(X[:, 0], X[:, 1], linestyle = '-', linewidth = 2, color = 'orange', alpha = 0.7,
                label = 'curve')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title('NICE parameterization')
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.12), ncol = 2)

        return fig, ax