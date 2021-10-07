from enum import auto
from load_data import tokenized_dataset
from optimizer import get_optimizer, get_scheduler
from utills import * 
from models import StratifiedSampler
from loss import get_loss

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import wandb
import os
import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def custom_train(config, model, train_dataset, valid_dataset, tokenizer): 
  optimizer = get_optimizer(model, config)
  scheduler = get_scheduler(optimizer, config)
  criterion = get_loss(config)

  wandb.watch(model)

  y = torch.from_numpy(np.array(train_dataset.labels))
  batch_sampler = StratifiedSampler(class_vector=y ,batch_size=config.batch_size)

  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=batch_sampler, num_workers=5)
  valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=5)
  
  os.makedirs(config.model_save_path, exist_ok=True)
  
  best_criterion = 0 # f1-score로
  early_count = 0

  for epoch in range(config.num_train_epochs):
    # your training routine
    train_loss, train_f1_score, train_auprc = train_per_epoch(config, train_loader, model, optimizer, criterion)

    # vlidation routine
    text_table = wandb.Table(columns=['pred_label', 'real_label', 'text'])
    valid_loss, valid_f1_score, valid_auprc = valid_per_epoch(config, valid_loader, model, criterion, text_table, valid_dataset, tokenizer)

    # learning rate controll
    scheduler.step()

    # wandb_logging
    logging_with_wandb(epoch, train_loss, train_f1_score, train_auprc, valid_loss, valid_f1_score, valid_auprc)

    # console_logging
    logging_with_console(epoch, train_loss, train_f1_score, train_auprc, valid_loss, valid_f1_score, valid_auprc)

    # save_best_model
    if valid_f1_score > best_criterion:
      best_criterion = valid_f1_score
      model.save_pretrained(config.model_save_path)

    if valid_f1_score < best_criterion:
      early_count += 1
      if config.early_stopping == early_count:
        break

    wandb.log({'Miss classification samples': text_table})
 
def train_per_epoch(config, train_loader, model, optimizer, criterion):
  model.train()
  scaler = GradScaler()

  pred_labels = []
  pred_probs = []
  target_labels = []
  train_loss = 0
  optimizer.zero_grad()
  for batch_idx, item in enumerate(tqdm(train_loader)):
    sentences = item['input_ids'].to(config.device)
    attention_mask = item['attention_mask'].to(config.device)
    target = item['labels'].to(config.device)

    with autocast():
      if config.use_entity_embedding:
        entity_embed = item['Entity_type_embedding'].to(config.device)
        entity_idxes = item['Entity_idxes'].to(config.device)
        pred = model.forward(sentences, attention_mask=attention_mask, entity_location=entity_idxes, entity_type_ids=entity_embed, labels=target)
      else:
        pred = model.forward(sentences, attention_mask=attention_mask, labels=target)
    logits = pred[1]

    loss = criterion(logits, target)
    scaler.scale(loss).backward()
    if batch_idx % 1 == 0:
      scaler.step(optimizer)#.step()
      scaler.update()
      optimizer.zero_grad()

    train_loss += loss.detach().cpu().numpy()
    pred_labels.extend(torch.argmax(logits.cpu(), dim=1).detach().cpu().numpy())
    pred_probs.extend(logits.detach().cpu().numpy())
    target_labels.extend(target.detach().cpu().numpy())

  train_loss /= batch_idx
  train_f1_score = klue_re_micro_f1(pred_labels, target_labels)
  train_auprc = klue_re_auprc(np.array(pred_probs), target_labels)

  return train_loss, train_f1_score, train_auprc

def valid_per_epoch(config, valid_loader, model, criterion, text_table, valid_dataset, tokenizer):
  with torch.no_grad():
    model.eval()

    pred_labels = []
    pred_probs = []
    target_labels = []
    valid_loss = 0

    for batch_idx, item in enumerate(tqdm(valid_loader)):
      sentences = item['input_ids'].to(config.device)
      attention_mask = item['attention_mask'].to(config.device)
      target = item['labels'].to(config.device)
      with autocast():
        if config.use_entity_embedding:
          entity_embed = item['Entity_type_embedding'].to(config.device)
          entity_idxes = item['Entity_idxes'].to(config.device)
          pred = model.forward(sentences, attention_mask=attention_mask, entity_location=entity_idxes, entity_type_ids=entity_embed, labels=target)
        else:
          pred = model.forward(sentences, attention_mask=attention_mask, labels=target)
      logits = pred[1]

      loss = criterion(logits, target)

      valid_loss += loss.detach().cpu().numpy()
      pred_labels.extend(torch.argmax(logits.cpu(), dim=1).detach().cpu().numpy())
      pred_probs.extend(logits.detach().cpu().numpy())
      target_labels.extend(target.detach().cpu().numpy())

    valid_loss /= batch_idx
    valid_f1_score = klue_re_micro_f1(pred_labels, target_labels)
    valid_auprc = klue_re_auprc(np.array(pred_probs), target_labels)

    count = 0
    for idx in range(len(target_labels)):
      if pred_labels[idx] != target_labels[idx]:
        ex_text = tokenizer.convert_tokens_to_string((tokenizer.convert_ids_to_tokens(valid_dataset[idx]['input_ids'], skip_special_tokens=True)))
        text_table.add_data(num_to_label([pred_labels[idx]])[0], num_to_label([target_labels[idx]])[0], ex_text)
        count += 1
      if count == 50:
        break

    return valid_loss, valid_f1_score, valid_auprc

