from optimizer import get_optimizer
from transformers.utils.dummy_pt_objects import get_scheduler
import torch
from loss import MyTrainer
from utils import * 

import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from load_data import *
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup


def train(config, model, train_dataset, valid_dataset): 
  #optimizer = get_optimizer(model, config)
  #scheduler = get_scheduler(optimizer, config)
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
  #optimizers = (optimizer, scheduler)
  

  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(                
                                                                    # enable logging to W&B
    output_dir=os.path.join(config.output_dir, config.run_path),    # output directory
    save_total_limit=config.save_total_limit,                       # number of total save model.
    save_steps=config.save_steps,                                   # model saving step.
    num_train_epochs=config.num_train_epochs,                       # total number of training epochs
    learning_rate=config.learning_rate,                             # learning_rate
    per_device_train_batch_size=config.batch_size,                  # batch size per device during training
    per_device_eval_batch_size=config.batch_size,                   # batch size for evaluation
    warmup_steps=config.warmup_steps,                               # number of warmup steps for learning rate scheduler
    weight_decay=config.weight_decay,                               # strength of weight decay
    logging_dir=os.path.join(config.logging_dir, config.run_path),  # directory for storing logs
    logging_steps=config.logging_steps,                             # log saving step.
    evaluation_strategy=config.evaluation_strategy,                 # evaluation strategy to adopt during training
                                                                    # `no`: No evaluation during training.
                                                                    # `steps`: Evaluate every `eval_steps`.
                                                                    # `epoch`: Evaluate every end of epoch.
    eval_steps = config.eval_steps,                                 # evaluation step.
    load_best_model_at_end = True,
    metric_for_best_model = 'micro f1 score',                                    # to define best model strategy
    seed = config.seed                                               # random seed
  )

  # Custom Loss 사용을 위해 Trainner 정의 (loss.py)
  trainer = MyTrainer(
    config=config,
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    #optimizers=optimizers,
    optimizers = (
      optimizer, get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps=500, num_training_steps=len(train_dataset)* config.num_train_epochs)
    ),
    compute_metrics=compute_metrics,      # define metrics function
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping)],
  )

  # train model
  trainer.train()
  model.save_pretrained(os.path.join(config.model_save_path, config.run_path))
