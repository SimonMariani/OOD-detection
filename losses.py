import torch
import torch.nn as nn
import torch.distributions as dist


class DPNKlLoss(nn.Module):

    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, alpha, labels):
        return dpn_kl_loss(alpha, labels, reduction=self.reduction)


class EDLMSELoss(nn.Module):

    def __init__(self, reduction='mean', annealing_step=100, epoch=1) -> None:
        super().__init__()
        self.reduction = reduction
        self.annealing_step = annealing_step
        self.epoch = epoch

    def forward(self, alpha, labels):
        return edl_mse_loss(alpha, labels, reduction=self.reduction, annealing_step=self.annealing_step,
                            epoch=self.epoch)


class DoubleLoss(nn.Module):

    def __init__(self, reduction='mean', criterion=None) -> None:
        super().__init__()
        self.reduction = reduction
        self.criterion = criterion

    def forward(self, logits, targets):  # logits should be a tuple of logits here
        print("here")
        print(self.criterion(logits[0], targets))
        print(self.criterion(logits[1], targets))
        return self.criterion(logits[0], targets) + self.criterion(logits[1], targets)


### The loss functions which are called by the loss classes, these can also be called directly ###
def dpn_kl_loss(alpha, labels, reduction='mean'):
    # We obtain the two dirichlet distributions and calculate the distance
    dirichlet_predict = dist.Dirichlet(alpha)
    dirichlet_target = dist.Dirichlet(labels)
    distance = dist.kl.kl_divergence(dirichlet_predict, dirichlet_target)

    # We aggregate the results
    return reduce(distance, reduction=reduction)


def edl_mse_loss(alpha, labels, annealing_step=100, epoch=1, reduction='mean'):
    S = torch.sum(alpha, dim=1, keepdims=True)  # the dirichlet strength
    pred = alpha / S  # the mean of the dirichlet, which is the same as the softmax if alpha = exp(logits)

    # TODO change to loss before (step back in equation)
    err = torch.sum((labels - pred) ** 2, dim=1, keepdims=True)
    var = torch.sum((pred * (1 - pred)) / (S + 1), dim=1,
                    keepdims=True)  # var = torch.sum( (alpha*(S-alpha) ) / (S*S*(S+1)), dim=1)
    mse = err + var

    # The regularization coefficient
    annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32),
                               torch.tensor(epoch / annealing_step, dtype=torch.float32))

    # Encode the alpha value for all the negative classes while keeping the true label value at 1
    alp = labels + (1 - labels) * alpha  # alp = evidence * (1 - labels) + 1
    kl = annealing_coef * dist.kl.kl_divergence(dist.Dirichlet(alp),
                                                dist.Dirichlet(torch.ones_like(alpha))).unsqueeze(1)  # kl_uniform(alp)
    loss = mse + kl

    # We aggregate the results
    return reduce(loss, reduction=reduction)


def reduce(tensor, reduction='mean'):
    if reduction == 'mean':
        return torch.mean(tensor, dim=0)
    elif reduction == 'sum':
        return torch.sum(tensor, dim=0)
    else:
        return tensor


# def kl_uniform(alpha):
#     ones = torch.ones((1, alpha.shape[1])).to(alpha.device)
#     precision = torch.sum(alpha, dim=1, keepdims=True)
#     num_classes = torch.sum(ones, dim=1, keepdims=True)
#
#     first = (torch.lgamma(precision)
#              - torch.sum(torch.lgamma(alpha), dim=1, keepdims=True)
#              + torch.sum(torch.lgamma(ones), dim=1, keepdims=True)
#              - torch.sum(torch.lgamma(num_classes), dim=1))
#
#     second = torch.sum((alpha - ones) * (torch.digamma(alpha) - torch.digamma(precision)), dim=1, keepdims=True)
#
#     return first + second

