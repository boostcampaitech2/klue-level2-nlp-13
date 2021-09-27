from transformers import Trainer, TrainingArguments
from utills import * 

def train(config, model, train_dataset): 
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    report_to="wandb",                              # enable logging to W&B
    run_name="bert-base-test",                   # Run name
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
    load_best_model_at_end = True 
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=train_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model/best.pt')
