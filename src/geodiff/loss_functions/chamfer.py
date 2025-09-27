import torch
import torch.nn as nn


class ChamferLoss(nn.Module):
    
    def __init__(self) -> None:
        super(ChamferLoss, self).__init__()
    
    
    def forward(self, Xc: torch.Tensor, Xt: torch.Tensor) -> torch.Tensor:
        # Compute pairwise squared Euclidean distances
        dists = torch.cdist(Xc, Xt, p = 2) ** 2  # (N, M)
        
        # Compute minimum distance for each point in Xc to Xt, and vice versa
        min_dist_Xc = torch.min(dists, dim = 1)[0]  # (N,)
        min_dist_Xt = torch.min(dists, dim = 0)[0]  # (M,)
        
        # Compute Chamfer distance
        loss = torch.mean(min_dist_Xc) + torch.mean(min_dist_Xt)
        return loss