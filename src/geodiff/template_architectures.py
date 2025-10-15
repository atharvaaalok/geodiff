import torch
from torch import nn


class MLP(nn.Module):
    r"""Flexible MLP: (Linear -> Norm -> Act) * layer_count, then Linear -> (optional out_act)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_count: int,
        hidden_dim: int,
        act_f = nn.GELU,
        norm_f = nn.BatchNorm1d,
        out_f = None,
    ) -> None:
        r"""
        Args:
            input_dim: Dimension of input vector to the network.
            output_dim: Dimension of output vector from the network.
            layer_count: Number of hidden blocks.
            hidden_dim: Number of neurons in each hidden layer.
            act_f: Activation function, provided as class. Default: `nn.PReLU`.
            norm_f: Normalization function, provided as class. Default: `nn.BatchNorm1d`.
            out_f: Output layer activation function, provided as class. Default: None
                (`nn.Identity`).
        """
        super().__init__()

        if act_f is None:
            act_f = nn.Identity
        if norm_f is None:
            norm_f = nn.Identity
        if out_f is None:
            out_f = nn.Identity

        # Save attributes in buffer so that they can be saved with state_dict
        self.register_buffer('input_dim', torch.tensor(input_dim, dtype = torch.int64))
        self.register_buffer('output_dim', torch.tensor(output_dim, dtype = torch.int64))
        self.register_buffer('layer_count', torch.tensor(layer_count, dtype = torch.int64))
        self.register_buffer('hidden_dim', torch.tensor(hidden_dim, dtype = torch.int64))


        # Define the network blocks
        self.blocks = nn.ModuleList()

        # Get the first block
        self.blocks.append(nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            norm_f(hidden_dim),
            act_f(),
        ]))
        # Add hidden blocks
        for i in range(layer_count - 1):
            self.blocks.append(nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                norm_f(hidden_dim),
                act_f(),
            ]))
        # Add output block
        self.blocks.append(nn.ModuleList([
            nn.Linear(hidden_dim, output_dim),
            out_f(),
        ]))


    def forward(self, X):
        for block in self.blocks:
            for layer in block:
                X = layer(X)

        return X


class ResMLP(nn.Module):
    r"""Flexible Residual MLP: (Linear -> Norm -> Act + residual) / 2 * layer_count, then Linear ->
    (optional out_act).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_count: int,
        hidden_dim: int,
        act_f = nn.GELU,
        norm_f = nn.BatchNorm1d,
        out_f = None,
    ) -> None:
        r"""
        Args:
            input_dim: Dimension of input vector to the network.
            output_dim: Dimension of output vector from the network.
            layer_count: Number of hidden blocks.
            hidden_dim: Number of neurons in each hidden layer.
            act_f: Activation function, provided as class. Default: `nn.PReLU`.
            norm_f: Normalization function, provided as class. Default: `nn.BatchNorm1d`.
            out_f: Output layer activation function, provided as class. Default: None
                (`nn.Identity`).
        """
        super().__init__()

        if act_f is None:
            act_f = nn.Identity
        if norm_f is None:
            norm_f = nn.Identity
        if out_f is None:
            out_f = nn.Identity

        # Save attributes in buffer so that they can be saved with state_dict
        self.register_buffer('input_dim', torch.tensor(input_dim, dtype = torch.int64))
        self.register_buffer('output_dim', torch.tensor(output_dim, dtype = torch.int64))
        self.register_buffer('layer_count', torch.tensor(layer_count, dtype = torch.int64))
        self.register_buffer('hidden_dim', torch.tensor(hidden_dim, dtype = torch.int64))


        # Define the network blocks
        self.blocks = nn.ModuleList()

        # Get the first block
        self.blocks.append(nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            norm_f(hidden_dim),
            act_f(),
        ]))
        # Add hidden blocks
        for i in range(layer_count - 1):
            self.blocks.append(nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                norm_f(hidden_dim),
                act_f(),
            ]))
        # Add output block
        self.blocks.append(nn.ModuleList([
            nn.Linear(hidden_dim, output_dim),
            out_f(),
        ]))


    def forward(self, X):
        for i, block in enumerate(self.blocks):
            # If for the first or last layer the dim does not match hidden dim, don't use residual
            # connection
            if i == 0:
                if self.input_dim != self.hidden_dim:
                    for layer in block:
                        X = layer(X)
                    continue
            elif i == len(self.blocks) - 1:
                if self.output_dim != self.hidden_dim:
                    for layer in block:
                        X = layer(X)
                    continue

            # In all other blocks use residual connection - this code is run if the input or output
            # dimension matches the hidden dimension
            residual = X
            for layer in block:
                X = layer(X)
            X = (X + residual) / 2.0

        return X