import matplotlib.pyplot as plt
import torch

from geodiff.aux_nets import PreAuxNet
from geodiff.nig_net import NIGnet
from geodiff.monotonic_nets import SmoothMinMaxNet
from geodiff.loss_functions.chamfer import ChamferLoss

from .utils import square, normalize_0_to_1


# Set the seed for reproducibility
torch.manual_seed(42)


# Get points on a square (curve to fit)
num_pts = 1000
X_square = square(num_pts)

# Normalize x values to the range [0, 1] to compare with other representation methods
X_square = normalize_0_to_1(X_square)


# Create a NIGnet object
# First create Pre-Aux and monotonic networks to pass to the NIGnet initializer
preaux_net = PreAuxNet(geometry_dim = 2, layer_count = 2, hidden_dim = 20)
monotonic_net = SmoothMinMaxNet(input_dim = 1, n_groups = 6, nodes_per_group = 6)
nig_net = NIGnet(
    geometry_dim = 2,
    layer_count = 4,
    preaux_net = preaux_net,
    monotonic_net = monotonic_net,
)


# Train the NIGnet parameters to fit the square
loss_fn = ChamferLoss()

learning_rate = 0.01
epochs = 1000
print_cost_every = 200

Y_train = X_square

optimizer = torch.optim.Adam(nig_net.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

for epoch in range(epochs):
    Y_model = nig_net(num_pts = num_pts)
    
    loss = loss_fn(Y_model, Y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    if epoch == 0 or (epoch + 1) % print_cost_every == 0:
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')


# Visualize the fitted NIGnet shape
fig, ax = nig_net.visualize(num_pts = num_pts)
plt.tight_layout()
plt.show()