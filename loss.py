import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.functional as F    

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class CrossEntropy_FoscalLoss(nn.Module):
    def __init__(self, class_weights, config):
        super().__init__()
        self.weight1 = config.loss1_weight
        self.weight2 = config.loss2_weight
        self.device = config.device
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(self.device, dtype=torch.float))(inputs, targets)
        fs_loss = FocalLoss()(inputs, targets)
        return ce_loss * self.weight1 + fs_loss * self.weight2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
        
class LabelSmoothingLoss(nn.Module):
    def __init__(self, config, classes=3, smoothing=0.05, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = config.num_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CrossEntropy_FoscalLoss_LabelSmoothingLoss(nn.Module):
    def __init__(self, class_weights, config):
        super().__init__()
        self.weight1 = config.loss1_weight
        self.weight2 = config.loss2_weight
        self.weight3 = config.loss3_weight
        self.device = config.device
        self.class_weights = class_weights
        self.config = config

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(self.device, dtype=torch.float))(inputs, targets)
        fs_loss = FocalLoss()(inputs, targets)
        ls_loss = LabelSmoothingLoss(self.config, self.config.num_classes)(inputs, targets)
        return ce_loss * self.weight1 + fs_loss * self.weight2 + ls_loss * self.weight3

def get_loss(config, class_weight):
    if config.loss == 'CrossEntropy':
        loss_func1 = torch.nn.CrossEntropyLoss()
    elif config.loss == 'Crossentropy_foscal':
        loss_func1 = CrossEntropy_FoscalLoss(class_weight, config)
    elif config.loss == 'CrossEntropy_weighted':
        loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to(config.device, dtype=torch.float))
    elif config.loss == 'Focal':
        loss_func1 = FocalLoss()
    elif config.loss == 'MSE':
        loss_func1 = MSELoss()
    elif config.loss == 'Crossentropy_focal_labelsmoothing':
        loss_func1 = CrossEntropy_FoscalLoss_LabelSmoothingLoss(class_weight, config)

    return loss_func1