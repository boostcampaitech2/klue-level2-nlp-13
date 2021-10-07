# KLUE - Relation Extraction
Team: CLUE (level2-nlp-13)

# Contents
1. [Requirements](#Requirements)
2. [Project files](#Project-files)
3. [Train](#Train)
4. [Inference](#Inference)

# 1. Requirements
```
torch == 1.9.0+cu102
transformers == 
sklearn == 0.24.2
numpy == 1.19.2
pandas == 1.1.5
```

# 2. Project files
* main.py - main process for training
* train.py - function for train with Trainer, Trainargumets class in transformers library
* custom_train.py - basic pipe line for pytorch model
* re_pretraining.py - re pretraining hugging face model with MLM task
* load_data.py - class and function for data load, tokenizing
* model.py - models for training
* mytokenizers.py - tokenizer
* optimizer.py - optimizer for training
* Loss.py - loss for training
* utills.py - Defining functions necessary for the overall process
* inference.py - inference for single model
* two_step_infernece - inference for dual model (binary model -> multi model)
* config.ini - Setting the necessary parameters for the overall learning process

# 3. Train
1. config.ini setting
```ini
[Path]
data_path = ./train.csv
test_data_path = ./test_data.csv
label_to_num = ./dict_label_to_num.pkl ; dict
num_to_label = ./dict_num_to_label.pkl ; dict
;need to modify
model_save_path = ./best_model/custom_roberta/
output_dir = ./results/custom_roberta/
logging_dir = ./logs/custom_roberta/
submission_file_name = custom_roberta.csv
augmentation_data_path = ./dataset/aug_data.csv

[Model]
;need to modify
model_name = any huggingface model name or defined model name
tokenizer_name = any huggingface model name
optimizer_name = AdamW
scheduler_name = CosineAnnealingLR
num_classes = 30
add_special_token = punct / punct_type / special
new_special_token_list = ['[E1]', '[/E1]', '[E2]', '[/E2]']
prediction_mode = all / binary / multi
use_entity_embedding = 1 ;1 is True, 0 is False

[Loss]
;need to modify
loss_name = Crossentropy_foscal
loss1_weight = 0.9
loss2_weight = 0.1

[Training]
num_train_epochs = 20
learning_rate = 5e-5
batch_size = 32
warmup_steps = 500
weight_decay = 0.01
early_stopping = 15
k_fold_num = 5
random_state = 42
use_aug_data = 0 ;1 is True, 0 is False

[Recording]
logging_steps = 100
save_total_limit = 1
save_steps = 500
evaluation_strategy = steps
eval_steps = 1

[WandB]
;need to modify
run_name = custom_roberta
project = <your project name>
entity = <your entity name>

[Inference] ; for two_step_inference.py
binary_model_path = ''
multi_model_path = ''
```

2. main execution
* -c is mandatory

```
main.py -c <config file path>
E.g. main.py -c config.ini
```

# 4. Inference
## 4.1 Single model
* -c is mandatory

```
inference.py -c <config file path>
```

## 4.2 Dual model

```
two_step_infernece.py -c <config file path>
```

# 5. Result
* final ranking: 
* Wandb  
![캡처](https://user-images.githubusercontent.com/72729802/136358624-51bd79f0-afd8-4e93-a5b6-e74a31a0afb2.PNG)