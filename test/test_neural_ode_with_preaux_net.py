import matplotlib.pyplot as plt
import torch

from geodiff.aux_nets import PreAuxNet
from geodiff.neural_ode import NeuralODE
from geodiff.loss_functions.chamfer import ChamferLoss
from geodiff.template_architectures import ResMLP
from geodiff.utils import sample_T

from .utils import square, normalize_0_to_1


# Set the seed for reproducibility
torch.manual_seed(42)


# Get points on a square (curve to fit)
num_pts = 1000
X_square = square(num_pts)
T = sample_T(geometry_dim = 2, num_pts = num_pts)

# Normalize x values to the range [0, 1] to compare with other representation methods
X_square = normalize_0_to_1(X_square)


# First create, fit and freeze a Pre-Aux net
preaux_net = PreAuxNet(geometry_dim = 2, layer_count = 2, hidden_dim = 20)

# Train the Pre-Aux net parameters to fit the square
loss_fn = ChamferLoss()

learning_rate = 0.01
epochs = 1000
print_cost_every = 200

Y_train = X_square

optimizer = torch.optim.Adam(preaux_net.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

print('Pre-Aux Net Training')
for epoch in range(epochs):
    Y_model = preaux_net(T)
    
    loss = loss_fn(Y_model, Y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    if epoch == 0 or (epoch + 1) % print_cost_every == 0:
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')
print('')


# Get the shape represented by Pre-Aux net before NeuralODE training
Y_model_before = preaux_net(T).detach().cpu()


# Freeze the Pre-Aux net
for param in preaux_net.parameters():
    param.requires_grad = False


# Create a NeuralODE object
# First create a torch ode network to pass to the NeuralODE initializer
geometry_dim = 2
ode_net = ResMLP(input_dim = geometry_dim + 1, output_dim = geometry_dim, layer_count = 2,
                 hidden_dim = 20, norm_f = torch.nn.LayerNorm, out_f = torch.nn.Tanh)
neural_ode = NeuralODE(
    geometry_dim = geometry_dim,
    ode_net = ode_net,
    preaux_net = preaux_net,
)


# Train the NeuralODE parameters to fit the square
loss_fn = ChamferLoss()

learning_rate = 0.01
epochs = 1000
print_cost_every = 200

Y_train = X_square

optimizer = torch.optim.Adam(neural_ode.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

print('NeuralODE Training')
for epoch in range(epochs):
    Y_model = neural_ode(num_pts = num_pts)
    
    loss = loss_fn(Y_model, Y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    if epoch == 0 or (epoch + 1) % print_cost_every == 0:
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')


# Get the shape represented by Pre-Aux net before NeuralODE training
Y_model_after = preaux_net(T).detach().cpu()

# Compare the shapes represented by Pre-Aux net before and after NeuralODE training
plt.plot(Y_model_before[:, 0], Y_model_before[:, 1], label = 'Before')
plt.plot(Y_model_after[:, 0], Y_model_after[:, 1], label = 'After')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pre-Aux Net: Before vs After')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()


# Visualize the fitted NeuralODE shape
fig, ax = neural_ode.visualize(num_pts = num_pts)
plt.tight_layout()
plt.show()