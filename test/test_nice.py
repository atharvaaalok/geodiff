import matplotlib.pyplot as plt
import torch

from geodiff.nice import NICE
from geodiff.loss_functions.chamfer import ChamferLoss
from geodiff.template_architectures import ResMLP

from .utils import square, normalize_0_to_1


# Get points on a square (curve to fit)
num_pts = 1000
X_square = square(num_pts)

# Normalize x values to the range [0, 1] to compare with other representation methods
X_square = normalize_0_to_1(X_square)


# Create a NICE object
# First create a coupling network to pass to the NICE initializer
coupling_net = ResMLP(input_dim = 1, output_dim = 1, layer_count = 2, hidden_dim = 20)
nice = NICE(
    geometry_dim = 2,
    layer_count = 4,
    coupling_net = coupling_net,
    preaux_net_layer_count = 2,
    preaux_net_hidden_dim = 20,
)


# Train the NICE parameters to fit the square
loss_fn = ChamferLoss()

learning_rate = 0.001
epochs = 1000
print_cost_every = 200

Y_train = X_square

optimizer = torch.optim.Adam(nice.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

for epoch in range(epochs):
    Y_model = nice(num_pts = num_pts)
    
    loss = loss_fn(Y_model, Y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    if epoch == 0 or (epoch + 1) % print_cost_every == 0:
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')


# Visualize the fitted NICE shape
fig, ax = nice.visualize(num_pts = num_pts)
plt.tight_layout()
plt.show()