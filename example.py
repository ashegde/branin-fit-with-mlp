import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from test_function import branin_function

#-------------------------------------------------------------

# simple MLP

class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        '''
        MLP Layer with GELU activations
        '''
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = True)
        self.act = nn.GELU()

    def forward(self, x: torch.tensor):
        return self.act(self.linear(x))

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_hidden: int):
        '''
        MLP with GeLU activations
        '''
        super(MLP, self).__init__()
        layers = [MLPLayer(in_features, hidden_features)] \
              + [MLPLayer(hidden_features, hidden_features) for ii in range(num_hidden)] \
              + [nn.Linear(hidden_features, out_features, bias = True)] 
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
  
#-----------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setting up training data
n = 5
x1_train = torch.linspace(-1, 1, n)
x2_train = torch.linspace(-1, 1, n)
X1_train, X2_train = torch.meshgrid(x1_train, x2_train)
X_train = torch.cat((X1_train[...,None], X2_train[...,None]), dim=-1)
Y_train = branin_function(X_train)[...,None]

x1_test = torch.linspace(-1, 1, 100)
x2_test = torch.linspace(-1, 1, 100)
X1_test, X2_test = torch.meshgrid(x1_test, x2_test)
X_test = torch.cat((X1_test[...,None], X2_test[...,None]), dim=-1)
Y_test = branin_function(X_test) [...,None]

X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)


hidden_features = 128
num_hidden = 5
model = MLP(in_features=2, hidden_features=hidden_features, out_features=1, num_hidden=num_hidden)
model.to(device)

weight_decay = 1e-5
learning_rate = 1e-4
# candidate parameters
param_dict = {pn: p for pn, p in model.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

# weight decay applied only to tensor weights of order >= 2 (i.e., not biases)
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {"params": decay_params, 'weight_decay': weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0}
        ]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
criterion = torch.nn.MSELoss(reduction='mean')

max_steps = 1000
losses = {"train": [], "test": []}

for step in range(max_steps):
    model.train()
    optimizer.zero_grad()

    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()

    losses["train"].append(loss.item())

    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        loss = criterion(pred, Y_test)
    losses["test"].append(loss.item())

    if step % 10 == 0:
        train_loss = losses["train"][-1]
        test_loss = losses["test"][-1]
        print(f"step {step} | train loss: {train_loss} | test_loss = {test_loss}")

# Plotting
model.eval()
with torch.no_grad():
    pred = model(X_test)

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3)


ax1.plot_surface(X1_test, X2_test, Y_test.squeeze(), cmap=cm.coolwarm, linewidth=0)
ax1.scatter(X1_train, X2_train, Y_train.squeeze(), c='r', s=1, alpha = 1.0, label='train')
ax1.set_title(f"Ground truth")
ax1.legend(loc="upper right")

#ax2.plot_surface(X1_test, X2_test, Y_test.squeeze(), cmap=cm.gray, linewidth=0, alpha=0.6)
ax2.scatter(X1_train, X2_train, Y_train.squeeze(), c='r', s=1, alpha = 1.0, label='train')
ax2.scatter(X1_test, X2_test, pred.squeeze(), c='b', s=1, alpha = 0.2, label = 'test')
ax2.set_title(f"MLP")
ax2.legend(loc="upper right")

ax3.plot(range(max_steps), losses["train"], 'r', label='train')
ax3.plot(range(max_steps), losses["test"], 'b', label='test')
ax3.set_title(f"Loss")
ax3.set_xlabel("Step")
ax3.set_yscale("log")
ax3.legend(loc="upper right")

fig.suptitle(f'Training')
plt.savefig(f"training_features-{hidden_features}_layers-{num_hidden}.png")
plt.close()