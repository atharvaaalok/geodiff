import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from geodiff.aux_nets import PreAuxNet


class RealNVP(nn.Module):
    r"""Real-valued Non-Volume Preserving (RealNVP) architecture as described in [1]_ pre-composed
    with a Pre-Auxilliary network.

    This module generates a closed, non-self-intersecting base geometry via a PreAuxNet and then
    applies a stack of RealNVP layers.

    .. math:
        TODO

    References:
        [1]: Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "Density estimation using real
            nvp." arXiv preprint arXiv:1605.08803 (2016).
    """

    def __init__(
        self,
        geometry_dim: int,
        layer_count: int,
        translation_net: nn.Module,
        scale_net: nn.Module,
        preaux_net_layer_count: int,
        preaux_net_hidden_dim: int,
        preaux_net_act_f = nn.GELU,
        preaux_net_norm_f = nn.BatchNorm1d,
        preaux_net_out_f = nn.Softplus,
        use_batchnormalization: bool = False,
        use_residual_connection: bool = True,
    ) -> None:
        r"""Initialize the RealNVP object.

        Args:
            geometry_dim: Dimension of the output geometry, e.g. 2d, 3d etc.
            layer_count: Number of hidden layers in the NICE net.
            translation_net: A torch network that is used as the coupling function.
            scale_net: A torch network that is used as the scaling function.
            preaux_net_layer_count: Number of hidden layers in the Pre-Aux net. If `None` then only
                closed transform is performed.
            preaux_net_hidden_dim: Number of neurons in each hidden layer of the Pre-Aux net.
            preaux_net_act_f: Activation function used in the Pre-Aux net.
            preaux_net_norm_f: Normalization function used in the Pre-Aux net.
            preaux_net_out_f: Output activation function used in the Pre-Aux net.
            use_batchnormalization: If `True`, applies batch normalization after each coupling
                layer.
            use_residual_connection: If `True`, applies a residual connection after each coupling
                layer.
        """
        super().__init__()

        # Save attributes in buffer so that they can be saved with state_dict
        self.register_buffer('geometry_dim', torch.tensor(geometry_dim, dtype = torch.int64))
        self.register_buffer('layer_count', torch.tensor(layer_count, dtype = torch.int64))
        if preaux_net_layer_count is not None:
            self.register_buffer('preaux_net_layer_count', torch.tensor(preaux_net_layer_count,
                                                                        dtype = torch.int64))
        else:
            self.register_buffer('preaux_net_layer_count', torch.tensor(-1, dtype = torch.int64))
        self.register_buffer('preaux_net_hidden_dim', torch.tensor(preaux_net_hidden_dim,
                                                                   dtype = torch.int64))
        self.register_buffer('use_batchnormalization', torch.tensor(use_batchnormalization,
                                                               dtype = torch.bool))
        self.register_buffer('use_residual_connection', torch.tensor(use_residual_connection,
                                                               dtype = torch.bool))
        

        # Create a Pre-Aux Net to close the shape
        self.preaux_net = PreAuxNet(
            geometry_dim = geometry_dim,
            layer_count = preaux_net_layer_count,
            hidden_dim = preaux_net_hidden_dim,
            act_f = preaux_net_act_f,
            norm_f = preaux_net_norm_f, 
            out_f = preaux_net_out_f
        )


        # Create a base mask to define the first partition
        d_partition = (geometry_dim // 2) if (geometry_dim % 2 == 0) else (geometry_dim + 1) // 2
        mask1 = torch.tensor([True] * d_partition + [False] * (geometry_dim - d_partition),
                                 dtype = torch.bool)
        mask2 = torch.tensor([False] * (geometry_dim - d_partition) + [True] * d_partition,
                                 dtype = torch.bool)
        self.register_buffer('mask1', mask1)
        self.register_buffer('mask2', mask2)


        # Create coupling nets for each NICE layer
        self.translation_net = nn.ModuleList()
        self.scale_net = nn.ModuleList()
        for i in range(layer_count):
            self.translation_net.append(copy.deepcopy(translation_net))
            self.scale_net.append(copy.deepcopy(scale_net))
        
        # Use batchnormalization with each coupling layer
        self.normalization_layers = nn.ModuleList()
        for i in range(layer_count):
            if use_batchnormalization:
                self.normalization_layers.append(nn.BatchNorm1d(geometry_dim))
            else:
                self.normalization_layers.append(nn.Identity())
        

        # Use scale parameters with each scaling net
        self.log_scale_parameters = nn.ParameterList()
        for i in range(layer_count):
            self.log_scale_parameters.append(nn.Parameter(torch.zeros(geometry_dim - d_partition)))


    def forward(self, T: torch.Tensor = None, num_pts: int = 100) -> torch.Tensor:
        r"""Compute the coordinates of the shape represented by the RealNVP object.

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
        T.to(device = next(self.parameters()).device)

        # First create a closed transform using the PreAux net
        X = self.preaux_net(T)

        # Apply the NICE coupling layers
        mask_switch = True
        for translation_net, scale_net, normalization_layer, log_scale in zip(
            self.translation_net,
            self.scale_net,
            self.normalization_layers,
            self.log_scale_parameters
        ):
            # Pick a mask for the partition
            mask = self.mask1 if mask_switch else self.mask2

            if self.use_residual_connection:
                residual = X

            # Create the partition
            x1 = X[:, mask]
            x2 = X[:, ~mask]

            # Compute the output of the additive coupling layer
            y1 = x1
            y2 = x2 * torch.exp(scale_net(x1)) * log_scale + translation_net(x1)

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

        return X


    def visualize(self, T: torch.Tensor = None, num_pts: int = 100, ax = None):
        r"""Plot geometry represented by the RealNVP object.

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
        ax.set_title('RealNVP parameterization')
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.12), ncol = 2)

        return fig, ax