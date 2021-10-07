import configparser
import wandb
import pandas as pd
import numpy as np
import json
import pickle as pickle
import ast
from collections import defaultdict

import torch
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
    config.augmentation_data_path = values['Path']['augmentation_data_path']

    # For Model
    config.model_name = values['Model']['model_name']
    config.tokenizer_name = values['Model']['tokenizer_name']
    config.optimizer_name = values['Model']['optimizer_name']
    config.scheduler_name = values['Model']['scheduler_name']
    config.num_classes = int(values['Model']['num_classes'])
    config.add_special_token = values['Model']['add_special_token']
    config.new_special_token_list = ast.literal_eval(values.get("Model", "new_special_token_list"))
    config.prediction_mode = values['Model']['prediction_mode']
    config.use_entity_embedding = values['Model']['use_entity_embedding']

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
    config.use_aug_data = values['Training'].getboolean('use_aug_data', 'b')

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
    
    # For Inference
    config.binary_model_path = values['Inference']['binary_model_path']
    config.multi_model_path = values['Inference']['multi_model_path']

    return config

def label_to_num(config, label):
  """
    라벨로 되어 있던 class를 숫자로 변환 합니다.
  """
  num_label = []
  with open(config.label_to_num, 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    if config.prediction_mode == 'multi':
      num_label.append(dict_label_to_num[v] - 1)
    else:
      num_label.append(dict_label_to_num[v])
  
  return num_label

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

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
    num_class = len(np.unique(labels))
    labels = np.eye(num_class)[labels]

    score = np.zeros((num_class,))
    for c in range(num_class):
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

# 각 클래스의 데이터 수 기반 class_weights 계산
def get_class_weights(train_label):
  _ , class_num = np.unique(train_label, return_counts = True)
  print("Class number: ", _)
  print("Class Balance: ", class_num)
  
  base_class = np.max(class_num)
  class_weight = (base_class / np.array(class_num))
  return class_weight

# wandb에 학습 결과 기록
def logging_with_wandb(epoch, train_loss, train_f1_score, train_auprc, valid_loss, valid_f1_score, valid_auprc):
  wandb.log({
    f"epoch": epoch,
    f"train_loss": train_loss,
    f"train_f1": train_f1_score,
    f"train_auprc": train_auprc,
    f"valid_loss": valid_loss,
    f"valid_f1": valid_f1_score,
    f"valid_auprc": valid_auprc,
    })

# console에 결과 출력
def logging_with_console(epoch, train_loss, train_f1_score, train_auprc, valid_loss, valid_f1_score, valid_auprc):
  print(f"epoch: {epoch} | "
        f"train_loss:{train_loss:.5f} | "
        f"train_f1:{train_f1_score:.2f} | "
        f"train_auprc:{train_auprc:.2f} | "
        f"valid_loss:{valid_loss:.5f} | "
        f"valid_f1:{valid_f1_score:.2f} | "
        f"valid_auprc:{valid_auprc:.2f}"
  )

# entity 표현 방식에 따른 entity 위치 계산
def get_entity_idxes(tokenizer, token_list, config):
  entity_embedding = np.zeros(len(token_list))
  if config.add_special_token == 'special':
    # 스페셜 토큰 위치로 쉽게 찾을 수 있음
    entity_embedding[np.where(token_list==32000)[0][0]+1:np.where(token_list==32001)[0][0]] = 1
    entity_embedding[np.where(token_list==32002)[0][0]+1:np.where(token_list==32003)[0][0]] = 1
    return entity_embedding
  elif config.add_special_token == 'punct_type':
    # @: 36, *: 14, +: 15, ^: 65, 사람: 3611, 단체: 3971, 기타: 5867, 장소: 4938, 수량: 12395, 날짜: 9384
    # 패턴을 이용해 찾기
    subj_1 = tokenizer.convert_tokens_to_ids('@')
    subj_2 = tokenizer.convert_tokens_to_ids('*')
    obj_1 = tokenizer.convert_tokens_to_ids('+')
    obj_2 = tokenizer.convert_tokens_to_ids('^')
    names = tokenizer.convert_tokens_to_ids(['사람','단체', '기타', '장소', '수량', '날짜'])

    sjb_start_idx = 0
    sjb_end_idx = 0
    for idx, t in enumerate(token_list):
        if t == subj_1 and token_list[idx+1] == subj_2 and (token_list[idx+2] in names):
            sjb_start_idx = idx + 4
            sjb_end_idx = sjb_start_idx + 1
            while token_list[sjb_end_idx] != subj_1:
                sjb_end_idx += 1
            break

    entity_embedding[sjb_start_idx:sjb_end_idx] = 1

    obj_start_idx = 0
    obj_end_idx = 0
    for idx, t in enumerate(token_list):
        if t == obj_1 and token_list[idx+1] == obj_2 and (token_list[idx+2] in names):
            obj_start_idx = idx + 4
            obj_end_idx = obj_start_idx + 1
            while token_list[obj_end_idx] != obj_1:
                obj_end_idx += 1
            break
    entity_embedding[obj_start_idx:obj_end_idx] = 2
    return entity_embedding, sjb_start_idx, sjb_end_idx, obj_start_idx, obj_end_idx

  elif config.add_special_token == 'punct':
    # 패턴을 이용해 찾기
    subj_1 = tokenizer.convert_tokens_to_ids('@')
    subj_2 = tokenizer.convert_tokens_to_ids('*')
    obj_1 = tokenizer.convert_tokens_to_ids('+')
    obj_2 = tokenizer.convert_tokens_to_ids('^')

    sjb_start_idx = 0
    sjb_end_idx = 0
    for idx, t in enumerate(token_list):
        if t == subj_1 and token_list[idx+1] == subj_2:
            sjb_start_idx = idx + 2
            sjb_end_idx = sjb_start_idx + 1
            while token_list[sjb_end_idx] != subj_1:
                sjb_end_idx += 1
            break

    entity_embedding[sjb_start_idx:sjb_end_idx] = 1

    obj_start_idx = 0
    obj_end_idx = 0
    for idx, t in enumerate(token_list):
        if t == obj_1 and token_list[idx+1] == obj_2:
            obj_start_idx = idx + 2
            obj_end_idx = obj_start_idx + 1
            while token_list[obj_end_idx] != obj_1:
                obj_end_idx += 1
            break

    return entity_embedding, sjb_start_idx, sjb_end_idx, obj_start_idx, obj_end_idx

# entity 표현 방식에 따른 entity 위치 계산한 것 반환 받아 dataset에 넣어주기
def insert_entity_idx_tokenized_dataset(tokenizer, dataset, config):
  if config.add_special_token == 'special':
      entity_embeddings = [get_entity_idxes(tokenizer, ids, config) for ids in dataset['input_ids'].numpy()]
      dataset['Entity_type_embedding'] = torch.tensor(entity_embeddings).to(torch.int64)
  elif config.add_special_token == 'punct' or config.add_special_token == 'punct_type':
      entity_embeddings = []
      entity_idxes = []
      for ids in dataset['input_ids'].numpy():
          entity_embedding, sjb_start_idx, sjb_end_idx, obj_start_idx, obj_end_idx = get_entity_idxes(tokenizer, ids, config)
          entity_embeddings.append(entity_embedding)
          entity_idxes.append([sjb_start_idx, sjb_end_idx, obj_start_idx, obj_end_idx])
      dataset['Entity_type_embedding'] = torch.tensor(entity_embeddings).to(torch.int64)
      dataset['Entity_idxes'] = torch.tensor(entity_idxes).to(torch.int64)

# joson 데이터 불러오기
def json_to_df():
    json_path = "/opt/ml/dataset/train/klue-re-v1.1_dev.json"
    with open(json_path) as f:
        json_object = json.load(f)
    data = defaultdict(list)
    for dict in json_object:
        for key, value in dict.items():
            data[key].append(str(value))

    df = pd.DataFrame(data)
    df = df.rename(columns={"guid": "id"})
    return df