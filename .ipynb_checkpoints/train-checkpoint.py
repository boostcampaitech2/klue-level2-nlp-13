from optimizer import get_optimizer
from transformers.utils.dummy_pt_objects import get_scheduler
import torch
from loss import MyTrainer
from utils import * 

import random
import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
import wandb

def train(config, model, train_dataset, valid_dataset): 
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    optimizers = (optimizer, scheduler)

    now_seed = random.randint(1,100)
    print("+"*7,now_seed,"+"*7)
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(                
                                                    # enable logging to W&B
        output_dir=config.output_dir,                   # output directory
        save_total_limit=config.save_total_limit,       # number of total save model.
        save_steps=config.save_steps,                   # model saving step.
        num_train_epochs=config.num_train_epochs,       # total number of training epochs
        learning_rate=config.learning_rate ,            # learning_rate
        per_device_train_batch_size=config.batch_size,  # batch size per device during training
        per_device_eval_batch_size=config.batch_size,   # batch size for evaluation
        warmup_steps=config.warmup_steps,               # number of warmup steps for learning rate scheduler
        weight_decay=config.weight_decay,               # strength of weight decay
        logging_dir=config.logging_dir,                 # directory for storing logs
        logging_steps=config.logging_steps,             # log saving step.
        evaluation_strategy=config.evaluation_strategy, # evaluation strategy to adopt during training
                                                        # `no`: No evaluation during training.
                                                        # `steps`: Evaluate every `eval_steps`.
                                                        # `epoch`: Evaluate every end of epoch.
        eval_steps = config.eval_steps,                 # evaluation step.
        load_best_model_at_end = True,
        seed = 516                           # random seed
    )

    # Custom Loss 사용을 위해 Trainner 정의 (loss.py)
    trainer = MyTrainer(
        config=config,
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        optimizers=optimizers,
        compute_metrics=compute_metrics      # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(config.model_save_path)

# def train(config, model, train_dataset, valid_dataset):
  
#   # load model and tokenizer
#   # MODEL_NAME = "bert-base-uncased"
#   # MODEL_NAME = "roberta-base"
#   MODEL_NAME = "klue/bert-base"
#   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#   # load dataset
#   train_dataset = load_data("../dataset/train/train.csv")
#   # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

#   train_label = label_to_num(train_dataset['label'].values)
#   # dev_label = label_to_num(dev_dataset['label'].values)

#   # tokenizing dataset
#   tokenized_train = tokenized_dataset(train_dataset, tokenizer)
#   # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

#   # make dataset for pytorch.
#   RE_train_dataset = RE_Dataset(tokenized_train, train_label)
#   # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

#   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#   print(device)
#   # setting model hyperparameter
#   model_config =  AutoConfig.from_pretrained(MODEL_NAME)
#   model_config.num_labels = 30

#   model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
#   print(model.config)
#   model.parameters
#   model.to(device)
  
#   # 사용한 option 외에도 다양한 option들이 있습니다.
#   # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
#   training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     save_total_limit=5,              # number of total save model.
#     save_steps=500,                 # model saving step.
#     num_train_epochs=20,              # total number of training epochs
#     learning_rate=5e-5,               # learning_rate
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=16,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     logging_steps=100,              # log saving step.
#     evaluation_strategy='steps', # evaluation strategy to adopt during training
#                                 # `no`: No evaluation during training.
#                                 # `steps`: Evaluate every `eval_steps`.
#                                 # `epoch`: Evaluate every end of epoch.
#     eval_steps = 500,            # evaluation step.
#     load_best_model_at_end = True 
#   )
#   trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=RE_train_dataset,         # training dataset
#     eval_dataset=RE_train_dataset,             # evaluation dataset
#     compute_metrics=compute_metrics         # define metrics function
#   )

#   # train model
#   trainer.train()
#   model.save_pretrained('./best_model')
# def main():
#   train()

# if __name__ == '__main__':
#   main()