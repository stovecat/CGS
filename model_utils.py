# Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

def get_criterion(creiterion_name, num_of_classes, device, num_of_class_samples=None):
    per_cls_weights = torch.ones(num_of_classes).to(device)
    if creiterion_name == 'CE':
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').to(device)
    elif creiterion_name== 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1.0, reduction='none').to(device)
    elif creiterion_name == 'LDAM':
        criterion = LDAMLoss(cls_num_list=num_of_class_samples, max_m=0.5, s=30, weight=per_cls_weights, reduction='none').to(device)
    return criterion

def focal_loss(input_values, gamma):
    """Computes the focal loss

    Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    """
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class FocalLoss(nn.Module):
    """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
    def __init__(self, weight=None, gamma=0., reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction), self.gamma)
    

class LDAMLoss(nn.Module):
    """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, reduction='mean'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        for i, num in enumerate(cls_num_list):
            if cls_num_list[i] == 0:
                m_list[i] = 0
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        print(m_list)
        self.m_list = m_list
        self.scale = s
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.scale * output, target, weight=self.weight, reduction=self.reduction)

