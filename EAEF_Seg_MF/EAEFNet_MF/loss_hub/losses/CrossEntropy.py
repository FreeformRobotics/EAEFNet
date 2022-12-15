import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

class CrossEntropyLoss(nn.Module):
    def __init__(
            self,class_weight=None,reduction='mean',avg_factor=None,ignore_index=-100,avg_non_ignore=False):
        """cross_entropy. The wrapper function for :func:`F.cross_entropy`
            Args:
                pred (torch.Tensor): The prediction with shape (N, 1).
                label (torch.Tensor): The learning label of the prediction.
                weight (torch.Tensor, optional): Sample-wise loss weight.
                    Default: None.
                class_weight (list[float], optional): The weight for each class.
                    Default: None.
                reduction (str, optional): The method used to reduce the loss.
                    Options are 'none', 'mean' and 'sum'. Default: 'mean'.
                avg_factor (int, optional): Average factor that is used to average
                    the loss. Default: None.
                ignore_index (int): Specifies a target value that is ignored and
                    does not contribute to the input gradients. When
                    ``avg_non_ignore `` is ``True``, and the ``reduction`` is
                    ``''mean''``, the loss is averaged over non-ignored targets.
                    Defaults: -100.
                avg_non_ignore (bool): The flag decides to whether the loss is
                    only averaged over non-ignored targets. Default: False.
                    `New in version 0.23.0.`
            """
        super().__init__()
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.avg_non_ignore = avg_non_ignore

    def forward(self,pred,label,weight=None):
        loss = F.cross_entropy(pred,label,weight=self.class_weight,reduction='none',ignore_index=self.ignore_index)
        if (self.avg_factor is None) and self.avg_non_ignore and self.reduction == 'mean':
            self.avg_factor = label.numel() - (label == self.ignore_index).sum().item()
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(loss, weight=weight, reduction=self.reduction, avg_factor=self.avg_factor)

        return loss
