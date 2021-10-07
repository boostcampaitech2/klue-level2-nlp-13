import torch
import torch.nn as nn
import torch.functional as F    
from transformers import Trainer

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
        if self.class_weights:
            ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(self.device, dtype=torch.float))(inputs, targets)
        else:
            ce_loss = nn.CrossEntropyLoss()(inputs, targets)
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
        self.device = config.device
        self.class_weights = class_weights
        self.config = config

    def forward(self, inputs, targets):
        if self.class_weights:
            ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(self.device, dtype=torch.float))(inputs, targets)
        else:
            ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        fs_loss = FocalLoss()(inputs, targets)
        ls_loss = LabelSmoothingLoss(self.config, self.config.num_classes)(inputs, targets)
        return ce_loss * self.weight1 + fs_loss * self.weight2 + ls_loss

# Custom Loss 사용을 위해 Trainner 정의 (loss.py)
class MyTrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def compute_loss(self, model, inputs, return_outputs=False):
        # config에 저장된 loss_name에 따라 다른 loss 계산
        if self.config.loss_name == 'CrossEntropy':
            custom_loss = torch.nn.CrossEntropyLoss()
        elif self.config.loss_name == 'Crossentropy_weighted_foscal':
            custom_loss = CrossEntropy_FoscalLoss(self.config.class_weight, self.config)
        elif self.config.loss_name == 'Crossentropy_foscal':
            custom_loss = CrossEntropy_FoscalLoss(None, self.config)
        elif self.config.loss_name == 'CrossEntropy_weighted':
            custom_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.config.class_weight).to(self.config.device, dtype=torch.float))
        elif self.config.loss_name == 'Focal':
            custom_loss = FocalLoss()
        elif self.config.loss_name == 'Crossentropy_weighted_focal_labelsmoothing':
            custom_loss = CrossEntropy_FoscalLoss_LabelSmoothingLoss(self.config.class_weight, self.config)
        elif self.config.loss_name == 'Crossentropy_weighted_focal_labelsmoothing':
            custom_loss = CrossEntropy_FoscalLoss_LabelSmoothingLoss(None, self.config)
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            loss = custom_loss(outputs[0], labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

def get_loss(config):
    if config.loss_name == 'CrossEntropy':
        custom_loss = torch.nn.CrossEntropyLoss()
    elif config.loss_name == 'Crossentropy_weighted_foscal':
        custom_loss = CrossEntropy_FoscalLoss(config.class_weight, config)
    elif config.loss_name == 'Crossentropy_foscal':
        custom_loss = CrossEntropy_FoscalLoss(None, config)
    elif config.loss_name == 'CrossEntropy_weighted':
        custom_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(config.class_weight).to(config.device, dtype=torch.float))
    elif config.loss_name == 'Focal':
        custom_loss = FocalLoss()
    elif config.loss_name == 'Crossentropy_focal_labelsmoothing':
        custom_loss = CrossEntropy_FoscalLoss_LabelSmoothingLoss(config.class_weight, config)
    
    return custom_loss
