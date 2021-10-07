#%%
import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig,AutoModelForMaskedLM, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
import argparse
import wandb
import importlib

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train(args):
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  # MODEL_NAME = "klue/roberta-base"

  MODEL_NAME = args.model_name
  RUN_NAME = args.run_name
  EPOCH = args.epochs
  BATCH_SIZE = args.batch_size
  MAX_LENGTH = args.max_length

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # csv to df & str to dict
  train_dataset = load_data("/opt/ml/dataset/train/train.csv")

  dev_json_path = "/opt/ml/KLUE-Baseline/data/klue_benchmark/klue-re-v1.1/klue-re-v1.1_dev.json"
  dev_dataset = json_to_df(dev_json_path)


  # make tokenized tensor, typed_entity_marker_punct
  module = importlib.import_module('load_data')
  train_entity_set = getattr(module, args.train_entity_set)
  dev_entity_set = getattr(module, args.dev_entity_set)

  tokenized_train, train_label = train_entity_set(train_dataset, tokenizer, MAX_LENGTH)
  tokenized_dev, dev_label = dev_entity_set(dev_dataset, tokenizer, MAX_LENGTH)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  print("train", len(RE_train_dataset))
  print("dev", len(RE_dev_dataset))


  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  #setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30
  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

  print(model.config)
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    
    num_train_epochs = EPOCH,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size = BATCH_SIZE,  # batch size per device during training
    per_device_eval_batch_size = 128,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 1,            # evaluation step.
    save_total_limit = 8,              # number of total save model.
    save_steps = 500,                 # model saving step.
    # save_strategy = "steps",

    load_best_model_at_end = True,    
    # report_to = ["wandb"],  # enable logging to W&B
    run_name = RUN_NAME,
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  
  model.save_pretrained(f'./best_model/{RUN_NAME}')


import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

  seed_everything(42)

  parser = argparse.ArgumentParser()

  # parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
  parser.add_argument('--epochs', type=int, default= 10, help='number of epochs to train (default: 30)')
  parser.add_argument('--batch_size', type=int, default= 24, help='input batch size for training (default: 64)')
  
  parser.add_argument('--max_length', type=int, default=200, help='')


  # parser.add_argument('--model_name', type=str, default='klue/bert-base', help='model type (default: ResNet18)')
  parser.add_argument('--model_name', type=str, default='klue/roberta-large', help='model type (default: ResNet18)')


  parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
  parser.add_argument('--save__dir', type=str, default='./best_model', help='model save at {SM_MODEL_DIR}/{name}')
  parser.add_argument('--run_name', required=True, type=str, default='name_nth_modelname', help='model name shown in wandb. (Usage: name_nth_modelname, Example: seyoung_1st_resnet18')
  
  parser.add_argument('--train_entity_set', required=True, type=str, default='entity_marker_punct')
  parser.add_argument('--dev_entity_set', required=True, type=str, default='entity_marker_punct')


  args = parser.parse_args()

  train(args)