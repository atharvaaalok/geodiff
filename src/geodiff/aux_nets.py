import torch
from torch import nn

from geodiff.transforms import closed_transform_2d, closed_transform_3d
from geodiff.template_architectures import MLP, ResMLP


class PreAuxNet(nn.Module):
    r"""Pre-Auxilliary Network (Pre-Aux Net) for nD that creates an initial simple closed manifold.

    In :math:`nD` it first maps :math:`[0, 1]^n` to a :math:`(n+1)`-sphere and then transforms that
    to a simple closed manifold using anisotropic radial scaling.
    """

    def __init__(
        self,
        geometry_dim: int,
        layer_count: int,
        hidden_dim: int,
        act_f = nn.GELU,
        norm_f = nn.BatchNorm1d,
        out_f = nn.Softplus,
    ) -> None:
        r"""Initialize the PreAuxNet object.

        Args:
            geometry_dim: Dimension of the output geometry, e.g. 2d, 3d etc.
            layer_count: Number of hidden layers in the Pre-Aux net. If `None` then only closed
                transform is performed.
            hidden_dim: Number of neurons in each hidden layer.
            act_f: Activation function used in the hidden layers.
            norm_f: Normalization function used in the hidden layers.
            out_f: Output activation function used in the Pre-Aux net.
        """
        super().__init__()

        # Save attributes in buffer so that they can be saved with state_dict
        self.register_buffer('geometry_dim', torch.tensor(geometry_dim, dtype = torch.int64))
        if layer_count is not None:
            self.register_buffer('layer_count', torch.tensor(layer_count, dtype = torch.int64))
        else:
            self.register_buffer('layer_count', torch.tensor(-1, dtype = torch.int64))
        self.register_buffer('hidden_dim', torch.tensor(hidden_dim, dtype = torch.int64))


        # Close the input domain edges to map to the same r, e.g. theta = 0, 2pi in 2D
        if geometry_dim == 2:
            self.closed_transform = closed_transform_2d
        elif geometry_dim == 3:
            self.closed_transform = closed_transform_3d

        if layer_count is None:
            self.forward_stack = nn.Identity()
        else:
            # Get the network mapping input to r
            self.forward_stack = ResMLP(
                input_dim = geometry_dim,
                output_dim = 1,
                layer_count = layer_count,
                hidden_dim = hidden_dim,
                act_f = act_f,
                norm_f = norm_f,
                out_f = out_f
            )
    

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        closed_manifold = self.closed_transform(T)
        r = self.forward_stack(closed_manifold)
        X = r * closed_manifold
        return X