import matplotlib.pyplot as plt
import torch

from geodiff.real_nvp import RealNVP
from geodiff.loss_functions.chamfer import ChamferLoss
from geodiff.template_architectures import ResMLP

from .utils import square, normalize_0_to_1


# Get points on a square (curve to fit)
num_pts = 1000
X_square = square(num_pts)

# Normalize x values to the range [0, 1] to compare with other representation methods
X_square = normalize_0_to_1(X_square)


# Create a NICE object
# First create the translation and scaling networks to pass to the RealNVP initializer
translation_net = ResMLP(input_dim = 1, output_dim = 1, layer_count = 2, hidden_dim = 20)
scale_net = ResMLP(input_dim = 1, output_dim = 1, layer_count = 2, hidden_dim = 20,
                   act_f = torch.nn.Tanh)
real_nvp = RealNVP(
    geometry_dim = 2,
    layer_count = 4,
    translation_net = translation_net,
    scale_net = scale_net,
    preaux_net_layer_count = 2,
    preaux_net_hidden_dim = 20,
)


# Train the RealNVP parameters to fit the square
loss_fn = ChamferLoss()

learning_rate = 0.001
epochs = 1000
print_cost_every = 200

Y_train = X_square

optimizer = torch.optim.Adam(real_nvp.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

for epoch in range(epochs):
    Y_model = real_nvp(num_pts = num_pts)
    
    loss = loss_fn(Y_model, Y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    if epoch == 0 or (epoch + 1) % print_cost_every == 0:
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')


# Visualize the fitted RealNVP shape
fig, ax = real_nvp.visualize(num_pts = num_pts)
plt.tight_layout()
plt.show()