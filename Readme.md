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
transformers == 4.10.0
sklearn == 0.24.1
numpy == 1.19.2
pandas == 1.3.3
```

# 2. Project files
* train.py - function for train with Trainer, Trainargumets class in transformers library
* load_data.py - class and function for data load, tokenizing
* inference.py - inference for single model
* re_pretraining.py - re pretraining hugging face model with MLM task

# 3. Train
1. parameter setting
```args.parser

[train.py]
parser.add_argument('--model_type', default="roberta",type=str, help='model type(default=bert)')
parser.add_argument('--model_name', default="klue/roberta-large",type=str, help='model name(default="klue/bert-base")')
parser.add_argument('--save_path', default="./",type=str, help='saved path(default=./)')
parser.add_argument('--save_step', default=500,type=int, help='model saving step(default=500)')
parser.add_argument('--save_limit', default=5,type=int, help='# of save model(default=5)')
parser.add_argument('--seed', type=int, default=20, help='random seed (default: 42)')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 20)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size per device during training (default: 16)')
parser.add_argument('--max_len', type=int, default=200, help='max length (default: 256)')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 5e-5)')
parser.add_argument('--weight_decay', type=float, default=0, help='strength of weight decay(default: 0.01)')
parser.add_argument('--warmup_steps', type=int, default=300, help='number of warmup steps for learning rate scheduler(default: 500)')
parser.add_argument('--warmup_ratio', type=float, default=0.2, help='number of warmup ratio for warmup steps of learning rate scheduler(default: 0.2)')
parser.add_argument('--scheduler', type=str, default="linear", help='scheduler(default: "linear")')

[inference.py]
parser.add_argument('--model_dir', type=str, default="/opt/ml/code/pretrained/opt/ml/kyunghyun/roberta-large-pretrained/roberta-large")
parser.add_argument('--model', type=str, default="klue/roberta-large")


- Default paramters -
[Path]
data_path = opt/ml/dataset/train/train.csv
test_data_path = opt/ml/dataset/test/test_data.csv
label_to_num = opt/ml/code/dict_label_to_num.pkl ; dict
num_to_label = opt/ml/code/dict_num_to_label.pkl ; dict
;need to modify
model_save_path = opt/ml/code/best_model/custom_roberta/
output_dir = opt/ml/code/results/custom_roberta/
logging_dir = opt/ml/code/logs/custom_roberta/
submission_file_name = roberta-large-{custom}.csv
augmentation_data_path = opt/ml/code/dataset/aug_data.csv

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
loss_name = Crossentropy
loss1_weight = 0.9
loss2_weight = 0.1

[Training]
num_train_epochs = 10
learning_rate = 5e-5
batch_size = 32
warmup_steps = 500
weight_decay = 0.0
random_state = 20
use_aug_data = 0 ;1 is True, 0 is False

[Recording]
logging_steps = 100
save_total_limit = 1
save_steps = 500
evaluation_strategy = steps
eval_steps = 1

[WandB]
;need to modify
run_name = <your run_name>
project = <your project name>
entity = <your entity name>

```

2. train execution
* help to search h.p(hyper parameter)

```

nohup sh -c 'python train.py {your option} && python train.py {your option} 1> /dev/null 2>&1' &
e.g) 
nohup sh -c 'python train.py --epochs 15 --max_len 200 && python train.py --epoch 30 --batch_size 16 1> /dev/null 2>&1' &

```

# 4. Inference
## 4.1 when using checkpoint
* --model_dir 

```
inference.py --model_dir "opt/ml/code/results/checkpointOOOO"
```

## 4.2 when using TAPT model
* --model_dir 
```
infernece.py -model_dir "/opt/ml/code/pretrained/opt/ml/kyunghyun/roberta-large-pretrained/roberta-large"
```

# 5. Result
* final ranking: 13/19 
* score public micro f1 72.940 / auprc 76.668 => private micro f1 71.910 / auprc 79.132
* Wandb  
![캡처](file:///var/folders/_c/b8720cx572xgmbsffhgf2m8h0000gn/T/TemporaryItems/(screencaptureui%E1%84%8B%E1%85%B5(%E1%84%80%E1%85%A1)%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%20%E1%84%8C%E1%85%A5%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%8C%E1%85%AE%E1%86%BC%202)/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-08%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.29.39.png)
