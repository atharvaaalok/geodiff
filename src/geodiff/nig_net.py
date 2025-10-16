import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from geodiff.aux_nets import PreAuxNet


class NIGnet(nn.Module):
    r"""Neural Injective Geometry network (NIGnet) architecture as described in [1]_.

    This module generates a closed, non-self-intersecting base geometry via a PreAuxNet and then
    applies a stack of NIGnet layers.

    .. math:
        TODO

    References:
        [1]: Atharva Aalok, Juan J. Alonso. 2025. NIGnets. https://github.com/atharvaaalok/NIGnets
    """

    available_intersection_modes = ['possible', 'impossible']

    def __init__(
        self,
        geometry_dim: int,
        layer_count: int,
        monotonic_net: nn.Module,
        preaux_net_layer_count: int,
        preaux_net_hidden_dim: int,
        preaux_net_act_f = nn.GELU,
        preaux_net_norm_f = nn.BatchNorm1d,
        preaux_net_out_f = nn.Softplus,
        use_batchnormalization: bool = False,
        use_residual_connection: bool = True,
        intersection_mode: str = 'possible',
    ) -> None:
        r"""Initialize the NIGnet object.

        Args:
            geometry_dim: Dimension of the output geometry, e.g. 2d, 3d etc.
            layer_count: Number of hidden layers in the NICE net.
            coupling_net: A torch monotonic network that is used as the monotonic function.
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
            intersection_mode: If `possible`, uses normal Linear layers, else if 'impossible' uses
                ExpLinear layers for guaranteed non-self-intersection.
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

        self.intersection_mode = intersection_mode


        # Create a Pre-Aux Net to close the shape
        self.preaux_net = PreAuxNet(
            geometry_dim = geometry_dim,
            layer_count = preaux_net_layer_count,
            hidden_dim = preaux_net_hidden_dim,
            act_f = preaux_net_act_f,
            norm_f = preaux_net_norm_f, 
            out_f = preaux_net_out_f
        )


        # Create linear layers and monotonic nets for each NIGnet layer
        linear_class = nn.Linear if intersection_mode == 'possible' else ExpLinear
        self.linear_layers = nn.ModuleList()
        self.monotonic_nets = nn.ModuleList()
        for i in range(layer_count):
            self.linear_layers.append(linear_class(geometry_dim, geometry_dim))
            self.monotonic_nets.append(copy.deepcopy(monotonic_net))


        # Use batchnormalization with each coupling layer
        self.normalization_layers = nn.ModuleList()
        for i in range(layer_count):
            if use_batchnormalization:
                self.normalization_layers.append(nn.BatchNorm1d(geometry_dim))
            else:
                self.normalization_layers.append(nn.Identity())


    def forward(self, T: torch.Tensor = None, num_pts: int = 1000) -> torch.Tensor:
        r"""Compute the coordinates of the shape represented by the NIGnet object.

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

        for linear_layer, monotonic_net in zip(self.linear_layers, self.monotonic_nets):
            if self.use_residual_connection:
                residual = X

            # Apply linear transformation
            X = linear_layer(X)

            # Apply monotonic network to each component of x separately
            if self.geometry_dim == 2:
                x1, x2 = X[:, 0:1], X[:, 1:2]
                X = torch.cat([monotonic_net(x1), monotonic_net(x2)], dim = 1)
            elif self.geometry_dim == 3:
                x1, x2, x3 = X[:, 0:1], X[:, 1:2], X[:, 2:3]
                X = torch.cat([monotonic_net(x1), monotonic_net(x2), monotonic_net(x3)], dim = 1)

            if self.use_residual_connection:
                # Apply residual connection
                X = (X + residual) / 2.0

        return X


    def visualize(self, T: torch.Tensor = None, num_pts: int = 1000, ax = None):
        r"""Plot geometry represented by the NIGnet object.

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
        ax.set_title('NIGnet parameterization')
        ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.12), ncol = 2)

        return fig, ax


class ExpLinear(nn.Module):
    """Linear layer with matrix-exponentiated weight transformation.

    Performs linear transformation using W = exp(weight_matrix) instead of direct weight matrix
    multiplication. This ensures injectivity of the transformation.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    W : torch.nn.Parameter
        The learnable weight matrix, which is exponentiated in the forward pass.
    bias : torch.nn.Parameter or None
        Optional learnable bias term for each output feature.

    Parameters
    ----------
    in_features : int
        Number of features in the input.
    out_features : int
        Number of features in the output.
    bias : bool, optional
        If True, a learnable bias is included, by default True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Initialize the ExpLinear module.

        Raises
        ------
        AssertionError
            If in_features != out_features.
        """

        super().__init__()

        assert in_features == out_features, 'ExpLinear requires in_features == out_features'

        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self) -> None:
        """
        Reset the parameters of the layer using Kaiming uniform initialization for the weight
        matrix, and a uniform distribution for the bias.
        """

        nn.init.kaiming_uniform_(self.W, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of ExpLinear.

        The weight matrix `self.W` is exponentiated using `torch.matrix_exp` before being applied to
        the input `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_features)
        """

        exp_weight = torch.matrix_exp(self.W)
        return F.linear(x, exp_weight.t(), self.bias)