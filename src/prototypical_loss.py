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


def prototypical_loss(input, target, n_support, margin=0.1):
    '''
    Compute the prototypical loss for a batch of samples.
    - margin: if > 0, adds a feasibility margin loss (Paper Source 6 aligned)
    '''
    classes = torch.unique(target)
    n_classes = len(classes)
    
    prototypes = []
    query_samples = []
    query_targets = []
    
    for i, c in enumerate(classes):
        all_indices = target.eq(c).nonzero(as_tuple=False).view(-1)
        # Support
        s_idxs = all_indices[:n_support]
        prototypes.append(input[s_idxs].mean(0))
        # Query
        q_idxs = all_indices[n_support:]
        query_samples.append(input[q_idxs])
        # Targets for these queries
        query_targets.append(torch.full((len(q_idxs),), i, device=input.device, dtype=torch.long))
    
    prototypes = torch.stack(prototypes) # [n_unique_classes, dim]
    query_samples = torch.cat(query_samples) # [n_total_query, dim]
    query_targets = torch.cat(query_targets) # [n_total_query]
    
    # Compute distances [n_total_query, n_unique_classes]
    dists = euclidean_dist(query_samples, prototypes)
    
    # Compute log probabilities
    log_p_y = F.log_softmax(-dists, dim=1)
    
    # Standard NLL Loss
    loss_val = F.nll_loss(log_p_y, query_targets)
    
    # ── Feasibility Margin Loss (Paper Source 6) ───────────────────────────
    # Encourages ground-truth class to be closer than the "best wrong" class by at least 'margin'
    if margin > 0:
        # scores = -distances (higher is better)
        scores = -dists # [n_query, n_classes]
        
        # gt_score: scores of the correct class
        gt_score = scores.gather(1, query_targets.unsqueeze(1)).squeeze(1)
        
        # best_wrong_score: max score among incorrect classes
        # Create mask for incorrect classes
        mask = torch.ones_like(scores).scatter_(1, query_targets.unsqueeze(1), 0.0)
        # Apply mask: keep incorrect scores, set correct one to very low value
        wrong_scores = scores * mask + (1 - mask) * (-1e9)
        best_wrong = wrong_scores.max(1)[0]
        
        # Hinge loss: max(0, best_wrong - gt_score + margin)
        feasibility_loss = torch.relu(best_wrong - gt_score + margin).mean()
        loss_val = loss_val + feasibility_loss

    _, y_hat = log_p_y.max(1)
    acc_val = y_hat.eq(query_targets).float().mean()

    return loss_val, acc_val
