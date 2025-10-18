import math

import matplotlib.pyplot as plt
import torch
from torch import nn

from geodiff.aux_nets import PreAuxNet
from geodiff.nice import NICE
from geodiff.loss_functions.chamfer import ChamferLoss
from geodiff.template_architectures import ResMLP

from .utils import square, rotate


# Set the seed for reproducibility
torch.manual_seed(42)


# Get points on a square (curve to fit)
num_pts = 1000
T = torch.linspace(0.0, 1.0, num_pts).reshape(-1, 1)
X_square = square(num_pts)


# Get target curves to fit using latent vectors
theta_1 = 30
theta_2 = -30
X1 = rotate(X_square, theta_1 * (math.pi / 180))
X2 = rotate(X_square, theta_2 * (math.pi / 180))
X_targets = [X1, X2]


# Set latent dimension and get latent codes
latent_dim = 8
latent_codes = nn.Embedding(num_embeddings = 2, embedding_dim = latent_dim)


# Create a NICE object
# First create Pre-Aux and coupling networks to pass to the NICE initializer
preaux_net = PreAuxNet(geometry_dim = 2, layer_count = 2, hidden_dim = 20, latent_dim = latent_dim,
                       norm_f = nn.LayerNorm)
coupling_net = ResMLP(input_dim = 1, output_dim = 1, layer_count = 2, hidden_dim = 20)
nice = NICE(
    geometry_dim = 2,
    layer_count = 4,
    preaux_net = preaux_net,
    coupling_net = coupling_net,
)


# Train the NICE parameters to fit the square
loss_fn = ChamferLoss()

learning_rate = 0.01
learning_rate_codes = 0.01
epochs = 1000
print_cost_every = 200

Y_train = X_square

optimizer = torch.optim.Adam([
    {'params': nice.parameters(), 'lr': learning_rate},
    {'params': latent_codes.parameters(), 'lr': learning_rate_codes},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

for epoch in range(epochs):
    loss = torch.tensor(0.0)

    for i, X in enumerate(X_targets):
        z_i = latent_codes.weight[i].reshape(1, -1)
        Y = nice(T = T, code = z_i)

        loss = loss + loss_fn(Y, X)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    if epoch == 0 or (epoch + 1) % print_cost_every == 0:
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')


# Visualize the shapes corresponding to the latent vectors
with torch.no_grad():
    z_1 = latent_codes.weight[0].reshape(1, -1)
    Y1 = nice(T = T, code = z_1).detach().cpu()

    z_2 = latent_codes.weight[1].reshape(1, -1)
    Y2 = nice(T = T, code = z_2).detach().cpu()

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

    ax1.plot(X1[:, 0], X1[:, 1], label = 'Original')
    ax1.plot(Y1[:, 0], Y1[:, 1], label = 'Latent')
    ax1.set_title(rf'$\theta$ = {theta_1}$^\circ$')
    ax1.legend()
    ax1.set_aspect('equal')

    ax2.plot(X2[:, 0], X2[:, 1], label = 'Original')
    ax2.plot(Y2[:, 0], Y2[:, 1], label = 'Latent')
    ax2.set_title(rf'$\theta$ = {theta_2}$^\circ$')
    ax2.legend()
    ax2.set_aspect('equal')

    plt.show()