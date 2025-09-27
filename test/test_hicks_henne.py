import matplotlib.pyplot as plt
import torch

from geodiff.hicks_henne import HicksHenne
from geodiff.loss_functions.chamfer import ChamferLoss

from .utils import circle, square, normalize_0_to_1


# Get points on a circle (baseline curve) and square (curve to fit)
num_pts = 1000
X_circle = circle(num_pts)
X_square = square(num_pts)

# For Hicks-Henne x values should lie in the range [0, 1]
X_circle = normalize_0_to_1(X_circle)
X_square = normalize_0_to_1(X_square)

# Extract upper and lower coordinates for Hicks-Henne
idx_upper = X_circle[:, 1] >= 0
idx_lower = X_circle[:, 1] < 0
X_upper_baseline = X_circle[idx_upper]
X_lower_baseline = X_circle[idx_lower]


# Create a HicksHenne object
hicks_henne = HicksHenne(
    X_upper_baseline = X_upper_baseline,
    X_lower_baseline = X_lower_baseline,
    polyexp_m = 5,
    polyexp_n_list = [0.2, 0.4, 0.6, 0.8],
    sin_m = 5,
    sin_n_list = None,
    sin_n_count = 12
)


# Train the Hicks-Henne parameters to fit the square
loss_fn = ChamferLoss()

learning_rate = 0.001
epochs = 1000
print_cost_every = 200

Y_train = X_square

optimizer = torch.optim.Adam(hicks_henne.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

for epoch in range(epochs):
    Y_model = hicks_henne()
    Y_model = torch.vstack([Y_model[0], Y_model[1]])
    
    loss = loss_fn(Y_model, Y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    if epoch == 0 or (epoch + 1) % print_cost_every == 0:
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')


# Visualize the fitted Hicks-Henne shape
fig, ax = hicks_henne.visualize()
plt.tight_layout()
plt.show()