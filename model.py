"""
MLP Model

This module contains a simple MLP model.
"""

import torch
from torch import nn


class MLPLayer(nn.Module):
    """
    Single layer of an MLP
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: nn.Module = nn.ReLU(),
    ):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation = activation

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward method
        """
        return self.activation(self.linear(x))


class MLP(nn.Module):
    """
    MLP architecture
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            num_hidden: int,
            activation: str,
    ):
        super(MLP, self).__init__()
        layers = [
            MLPLayer(
                in_features,
                hidden_features,
                nn.ReLU() if activation.lower() == 'relu' else nn.GELU(),
            ),
            *[
                MLPLayer(
                    hidden_features,
                    hidden_features,
                    nn.ReLU() if activation.lower() == 'relu' else nn.GELU(),
                ) for ii in range(num_hidden)
            ],
            nn.Linear(
                hidden_features,
                out_features,
                bias=True,
            ),
        ]
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                # nn.init.zeros_(module.bias)
                nn.init.normal_(module.bias)

# class MLP(nn.Module):
#     """
#     MLP architecture
#     """
#     def __init__(
#             self,
#             in_features: int,
#             hidden_features: int,
#             out_features: int,
#             num_hidden: int,
#             activation: str,
#     ):
#         super(MLP, self).__init__()
#         layers = [MLPLayer(in_features, hidden_features, activation)]
#         layers.extend([
#             MLPLayer(hidden_features,
#                      hidden_features,
#                      activation,
#                      ) for ii in range(num_hidden)
#         ])
#         layers.extend([nn.Linear(hidden_features, out_features, bias=True)])
#         self.net = nn.Sequential(*layers)

#         self.apply(self._init_weights)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.kaiming_normal_(module.weight)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)


