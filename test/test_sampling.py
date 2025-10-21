import matplotlib.pyplot as plt
import torch

from geodiff.utils import sample_T


num_pts = 10000
geometry_dim = 3
device = 'cpu'


T = sample_T(geometry_dim = geometry_dim, num_pts = num_pts, device = device)
print(f'{T.shape = }')
print(f'{T.device = }')


# Visualize the sampled points in the parameter domain
plt.scatter(T[:, 0], T[:, 1], s = 6, alpha = 0.85, linewidths = 0)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.xlabel('t'); plt.ylabel('s')
plt.title('Farthest Point Sampling')
plt.tight_layout()
plt.show()