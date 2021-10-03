import configparser
import wandb
import numpy as np
import pickle as pickle
import ast
import torch
import random

import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def read_config(paths):
  config = wandb.config

  values = configparser.ConfigParser()
  values.read(paths, encoding='utf-8')

  # For Path
  config.data_path = values['Path']['data_path']
  config.model_save_path = values['Path']['model_save_path']
  config.test_data_path = values['Path']['test_data_path']
  config.label_to_num = values['Path']['label_to_num']
  config.num_to_label = values['Path']['num_to_label']
  config.output_dir = values['Path']['output_dir']
  config.logging_dir = values['Path']['logging_dir']
  config.submission_file_name = values['Path']['submission_file_name']

  # For Model
  config.model_name = values['Model']['model_name']
  config.tokenizer_name = values['Model']['tokenizer_name']
  config.optimizer_name = values['Model']['optimizer_name']
  config.scheduler_name = values['Model']['scheduler_name']
  config.num_classes = int(values['Model']['num_classes'])
  config.add_special_token = values['Model'].getboolean('add_special_token', 'b')
  #config.new_special_token_list = ast.literal_eval(values.get("Model", "new_special_token_list"))

  # For Loss
  config.loss_name = values['Loss']['loss_name']
  config.loss1_weight = float(values['Loss']['loss1_weight'])
  config.loss2_weight = float(values['Loss']['loss2_weight'])

  # For Training
  config.num_train_epochs = int(values['Training']['num_train_epochs'])
  config.learning_rate = float(values['Training']['learning_rate'])
  config.batch_size = int(values['Training']['batch_size'])
  config.warmup_steps = int(values['Training']['warmup_steps'])
  config.weight_decay = float(values['Training']['weight_decay'])
  config.early_stopping = int(values['Training']['early_stopping'])
  config.k_fold_num = int(values['Training']['k_fold_num'])
  config.random_state = int(values['Training']['random_state'])

  # For Recording
  config.logging_steps = int(values['Recording']['logging_steps'])
  config.save_total_limit = int(values['Recording']['save_total_limit'])
  config.save_steps = int(values['Recording']['save_steps'])
  config.evaluation_strategy = values['Recording']['evaluation_strategy']
  config.eval_steps = int(values['Recording']['eval_steps'])

  # For WandB
  config.run_name = values['WandB']['run_name']
  config.project = values['WandB']['project']
  config.entity = values['WandB']['entity']

  # Etc
  config.seed = int(values['Etc']['seed'])

  return config

def label_to_num(config, label):
  num_label = []
  with open(config.label_to_num, 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])

  return num_label

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

def get_class_weights(train_label):
  _ , class_num = np.unique(train_label, return_counts = True)
  print("Class number: ", _)
  print("Class Balance: ", class_num)

  base_class = np.max(class_num)
  class_weight = (base_class / np.array(class_num))
  return class_weight

def seed_everything(seed):
  """To make fixed random state

  Args:
      seed ([int]): [seed key number]
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)  