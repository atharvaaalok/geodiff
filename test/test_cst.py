import matplotlib.pyplot as plt
import torch

from geodiff.cst import CST
from geodiff.loss_functions.chamfer import ChamferLoss

from .utils import square, normalize_0_to_1


# Get points on a square (curve to fit)
num_pts = 1000
X_square = square(num_pts)

# For CST x values should lie in the range [0, 1]
X_square = normalize_0_to_1(X_square)


# Create a CST object
cst = CST(
    n1 = 0.5,
    n2 = 0.5,
    upper_basis_count = 12,
    lower_basis_count = 12,
    upper_te_thickness = 0,
    lower_te_thickness = 0
)


# Train the CST parameters to fit the square
loss_fn = ChamferLoss()

learning_rate = 0.001
epochs = 1000
print_cost_every = 200

Y_train = X_square

optimizer = torch.optim.Adam(cst.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

for epoch in range(epochs):
    Y_model = cst(num_pts = num_pts)
    Y_model = torch.vstack([Y_model[0], Y_model[1]])
    
    loss = loss_fn(Y_model, Y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    if epoch == 0 or (epoch + 1) % print_cost_every == 0:
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')


# Visualize the fitted CST shape
fig, ax = cst.visualize(num_pts = num_pts)
plt.tight_layout()
plt.show()