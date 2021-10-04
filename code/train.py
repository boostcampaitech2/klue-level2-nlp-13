import sys
sys.path.insert(0, '/opt/ml/code/')

import pickle as pickle
import os
import random
import argparse

import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification , XLMRobertaConfig

from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers import ElectraTokenizer , ElectraModel , ElectraConfig
from transformers.utils.dummy_pt_objects import DistilBertForSequenceClassification

from transformers import BertForSequenceClassification, XLMRobertaForSequenceClassification, ElectraForSequenceClassification, DistilBertForSequenceClassification

from load_data import *

from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import wandb , logging


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
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

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

#------* Fix Seeds * -----------#
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#------* get logger * -----------#
def __get_logger():
    """로거 인스턴스 반환 """

    __logger = logging.getLogger('logger')

    # # 로그 포멧 정의
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger



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




def train():

  logger = __get_logger()

  logger.info("*************SETTING************")
  logger.info(f"model_name : {args.model_name}")
  logger.info(f"seed : {args.seed}")
  logger.info(f"scheduler : {args.scheduler}")
  logger.info(f"epochs : {args.epochs}")
  logger.info("********************************\n")

  # 1. Start a new run
  wandb.init(project='LJH', entity='clue')

  #fix a seed
  seed_everything(args.seed)
  # load model and tokenizer

  MODEL_NAME = args.model_name
  tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
  #tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("/opt/ml/dataset/train/added_data.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv")
  #train_dataset , dev_dataset = train_test_split(train_dataset, test_size = 0.1 , random_state= 42)

  train_label = label_to_num(train_dataset['label'].values)
  #dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
 
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  
  model_config.num_labels = 30

  model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
   
  print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir= args.save_path + 'results',          # output directory
    save_total_limit=args.save_limit,              # number of total save model.
    save_steps=args.save_step,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir=args.save_path + 'logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    lr_scheduler_type=args.scheduler,            # scheduler
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True ,
    #------* Wandb * -----------#
    report_to="wandb",  # enable logging to W&B
    run_name="koelectra-LJH"  # name of the W&B run (optional)
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_train_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')

def main():
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', default="bert",type=str, help='model type(default=bert)')
  parser.add_argument('--model_name', default="klue/bert-base",type=str, help='model name(default="klue/bert-base")')
  parser.add_argument('--save_path', default="./",type=str, help='saved path(default=./)')
  parser.add_argument('--save_step', default=500,type=int, help='model saving step(default=500)')
  parser.add_argument('--save_limit', default=5,type=int, help='# of save model(default=5)')
  parser.add_argument('--seed', type=int, default=1, help='random seed (default: 42)')
  parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 20)')
  parser.add_argument('--batch_size', type=int, default=16, help='batch size per device during training (default: 16)')
  parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 5e-5)')
  parser.add_argument('--weight_decay', type=float, default=0.01, help='strength of weight decay(default: 0.01)')
  parser.add_argument('--warmup_steps', type=int, default=500, help='number of warmup steps for learning rate scheduler(default: 500)')
  parser.add_argument('--scheduler', type=str, default="linear", help='scheduler(default: "linear")')
  args = parser.parse_args()
  print(args)

  # 2. Save model inputs and hyperparameters
  config = wandb.config
  config.learning_rate = args.lr
  config.epochs = args.epochs
  config.model_name = args.model_name
  config.scheduler = args.scheduler
  config.batch_size = args.batch_size

  #fix a seed
  seed_everything(args.seed)

  main()
    


