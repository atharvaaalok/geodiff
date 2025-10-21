import matplotlib.pyplot as plt
import torch

from geodiff.utils import sample_T
from geodiff.transforms import closed_transform_3d


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


# Use the closed transform to compute points on the initial closed manifold
closed_manifold = closed_transform_3d(T)

fig, ax = plt.subplots(subplot_kw = {'projection': '3d'})
ax.scatter(closed_manifold[:, 0], closed_manifold[:, 1], closed_manifold[:, 2], s = 6,
           alpha = 0.85, linewidths = 0)
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1]);
ax.set_box_aspect([1, 1, 1])
ax.set_title('Farthest Point Sampling with Uniform Closed Transform')
plt.tight_layout()
plt.show()