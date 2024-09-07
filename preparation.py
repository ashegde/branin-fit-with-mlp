"""
Data preparation

This module provides functionality for setting up the training
and test datasets.
"""

import torch

from test_function import branin_function


def prepare_data(
        num_train_gridpoints: int,
        num_test_gridpoints: int,
        device: torch.device,
) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Prepares training and test datasets

    Args:
        num_train_gridpoints (int): Number of training gridpoints.
        num_test_gridpoints (int): Number of testing gridpoints.
        device (str): Device on which to place tensor.

    Returns:
        tuple: Tensors for train and test inputs/outputs
    """
    train_x1 = torch.linspace(-1, 1, num_train_gridpoints)
    train_x2 = torch.linspace(-1, 1, num_train_gridpoints)
    train_x1, train_x2 = torch.meshgrid(train_x1, train_x2)
    train_x = torch.cat(
        (
            train_x1[..., None],
            train_x2[..., None],
        ),
        dim=-1,
    )
    train_y = branin_function(train_x)[..., None]

    test_x1 = torch.linspace(-1, 1, num_test_gridpoints)
    test_x2 = torch.linspace(-1, 1, num_test_gridpoints)
    test_x1, test_x2 = torch.meshgrid(test_x1, test_x2)
    test_x = torch.cat(
        (
            test_x1[..., None],
            test_x2[..., None],
        ),
        dim=-1,
    )
    test_y = branin_function(test_x)[..., None]

    return (
        train_x.to(device),
        train_y.to(device),
        test_x.to(device),
        test_y.to(device),
    )
