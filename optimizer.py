import torch
from transformers import AdamW

def get_optimizer(model, config):
    if config.optimizer_name == 'AdamW':
        optimizer = AdamW(model.parameters(), lr = config.learning_rate)
    elif config.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    return optimizer

def get_scheduler(optimizer, config):
    if config.scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10,
            eta_min = 0  
        )
    elif config.scheduler_name == 'LRscheduler':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                        lr_lambda=lambda epoch: 0.95 ** epoch,
                        last_epoch=-1,
                        verbose=False)

    return scheduler