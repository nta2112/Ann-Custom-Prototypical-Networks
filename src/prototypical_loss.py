# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Compute the prototypical loss for a batch of samples.
    If the batch contains multiple episodes, they are treated as a single large N-way task,
    which is a common optimization (higher-way training).
    '''
    device = input.device
    
    # Get unique classes in the batch
    classes = torch.unique(target)
    n_classes = len(classes)
    
    # Calculate n_query (assuming all classes have same number of samples)
    n_query = target.eq(classes[0].item()).sum().item() - n_support
    
    # Extract support and query indices for each class
    support_idxs = []
    query_idxs = []
    for c in classes:
        all_indices = target.eq(c).nonzero(as_tuple=False).view(-1)
        support_idxs.append(all_indices[:n_support])
        query_idxs.append(all_indices[n_support:])
    
    # Calculate prototypes (barycentres)
    prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs]) # [n_classes, dim]
    
    # Calculate query samples
    query_idxs = torch.cat(query_idxs)
    query_samples = input[query_idxs] # [n_classes * n_query, dim]
    
    # Compute distances [n_classes * n_query, n_classes]
    dists = euclidean_dist(query_samples, prototypes)
    
    # Compute log probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    
    # Target indices for each query sample
    target_inds = torch.arange(0, n_classes, device=device).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()
    
    # Loss and Accuracy
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val, acc_val
