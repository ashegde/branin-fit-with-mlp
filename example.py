"""
In this module, we explore fitting a branin function with ReLU and GeLU-based
MLPs. Specifically, we compare fits of the following form:

(1) ReLU activations
(2) GeLU activations
"""
import argparse

import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch import nn, optim

from model import MLP
from preparation import prepare_data
from test_function import branin_function


def main(args: argparse):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_features = 128
    num_hidden = 5
    model = MLP(
        in_features=2,
        hidden_features=hidden_features,
        out_features=1,
        num_hidden=num_hidden,
        activation=args.activation,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.MSELoss(reduction='mean')

    train_x, train_y, test_x, test_y = prepare_data(
        args.num_train_gridpoints,
        args.num_test_gridpoints,
        device,
    )
    losses = {'train': [], 'test': []}

    for step in range(args.max_steps):
        model.train()
        optimizer.zero_grad()

        pred = model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

        losses["train"].append(loss.item())

        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            loss = criterion(pred, test_y)
        losses["test"].append(loss.item())

        if step % 10 == 0:
            train_loss = losses["train"][-1]
            test_loss = losses["test"][-1]
            print(
                f'step {step} |',
                f'train loss: {train_loss} |',
                f'test_loss = {test_loss}',
            )

    # Plotting

    model.eval()
    with torch.no_grad():
        pred = model(test_x)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.plot_surface(
        test_x[:, :, 0],
        test_x[:, :, 1],
        test_y.squeeze(),
        cmap=cm.coolwarm,
        linewidth=0,
    )
    ax1.scatter(
        train_x[:, :, 0],
        train_x[:, :, 1],
        train_y.squeeze(),
        c='r', s=1, alpha=1.0, label='train',
    )
    ax1.set_title('Ground truth')
    ax1.legend(loc='upper right')

    #ax2.plot_surface(test_x1, train_x2, test_y.squeeze(), cmap=cm.gray, linewidth=0, alpha=0.6)
    ax2.scatter(
        train_x[:, :, 0],
        train_x[:, :, 1],
        train_y.squeeze(),
        c='r', s=1, alpha=1.0, label='train',
    )
    ax2.scatter(
        test_x[:, :, 0],
        test_x[:, :, 1],
        pred.squeeze(),
        c='b', s=1, alpha=0.2, label='test',
    )
    ax2.set_title(f'MLP-{args.activation}')
    ax2.legend(loc='upper right')

    ax3.plot(
        range(args.max_steps),
        losses["train"],
        'r', label='train',
    )
    ax3.plot(
        range(args.max_steps),
        losses["test"],
        'b', label='test',
    )
    ax3.set_title('Loss')
    ax3.set_xlabel('Step')
    ax3.set_yscale('log')
    ax3.legend(loc='upper right')

    fig.suptitle(f'Example with {args.activation} activations')
    plt.savefig(
        f'training_activation-{args.activation}_features-{hidden_features}_layers-{num_hidden}.png',
        dpi=300,
    )
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run example')
    parser.add_argument(
        '--num_train_gridpoints',
        type=int,
        default=5,
        help='number of training gridpoints per dim',
    )
    parser.add_argument(
        '--num_test_gridpoints',
        type=int,
        default=100,
        help='number of testing gridpoints per dim',
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        help='activations: relu or gelu?',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='learning rate',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='weight decay',
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='maximum steps for optimization',
    )
    args = parser.parse_args()
    main(args)
