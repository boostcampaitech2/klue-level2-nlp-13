import pandas as pd
import numpy as np
import torch
from ast import literal_eval

from utills import *

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels, config):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.config = config

  def __getitem__(self, idx):
    # 32000 [E1] // 32001 [/E1] // 32002 [E2] // 32003 [/E2]
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entities = []
  subject_entity_types = []
  subject_idx = []

  object_entities = []
  object_entity_types = []
  object_idx = []
  for subject_entity, object_entity in zip(dataset['subject_entity'], dataset['object_entity']):
    subject_entity = literal_eval(subject_entity)
    object_entity = literal_eval(object_entity)

    subject_word = subject_entity['word']
    object_word = object_entity['word']

    subj_type = subject_entity['type']
    obj_type = object_entity['type']

    subject_s_idx = int(subject_entity['start_idx'])
    subject_e_idx = int(subject_entity['end_idx'])

    object_s_idx = int(object_entity['start_idx'])
    object_e_idx = int(object_entity['end_idx'])
    
    subject_entities.append(subject_word)
    subject_entity_types.append(subj_type)
    subject_idx.append([subject_s_idx, subject_e_idx])

    object_entities.append(object_word)
    object_entity_types.append(obj_type)
    object_idx.append([object_s_idx, object_e_idx])

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],
          'subject_entity':subject_entities, 'subject_entity_types': subject_entity_types, 'subject_entity_idx': subject_idx,
          'object_entity':object_entities, 'object_entity_types': object_entity_types, 'object_entity_idx': object_idx,'label':dataset['label'],})

  return out_dataset

def load_data(dataset_dir, config, type):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)

  if config.prediction_mode == 'binary':
    pd_dataset.loc[pd_dataset.label != 'no_relation', 'label'] = 'org:top_members/employees'
  elif config.prediction_mode == 'multi':
    pd_dataset = pd_dataset.loc[pd_dataset.label != 'no_relation']

  dataset = preprocessing_dataset(pd_dataset)
  if type == 'main':
    valid = preprocessing_dataset(json_to_df())
    return dataset, valid
  return dataset

def tokenized_dataset(config, dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  
  concat_entity = []
  if config.add_special_token == 'punct_type':
    type_to_ko = {"PER":"사람", "ORG":"단체", "POH" : "기타", "LOC" : "장소", "NOH" : "수량", "DAT" : "날짜"}
    for idx, (e01_idx, e01_type, e02_idx, e02_type) in enumerate(zip(dataset['subject_entity_idx'], dataset['subject_entity_types'], dataset['object_entity_idx'], dataset['object_entity_types'])):
        temp_sentence = dataset['sentence'].iloc[idx]
        temp_sentence = (temp_sentence[:e01_idx[0]] + f'@ * {type_to_ko[e01_type]} * ' + temp_sentence[e01_idx[0]:e01_idx[1]+1] + ' @ ' 
        + temp_sentence[e01_idx[1]+1:e02_idx[0]] + f'+ ^ {type_to_ko[e02_type]} ^ ' + temp_sentence[e02_idx[0]:e02_idx[1]+1] + ' + ' + temp_sentence[e02_idx[1]+1:])
        dataset['sentence'].iloc[idx] = temp_sentence
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        #return_token_type_ids=False, # 문장 id
        )
  elif config.add_special_token == 'punct':
    for idx, (e01_idx, e01_type, e02_idx, e02_type) in enumerate(zip(dataset['subject_entity_idx'], dataset['subject_entity_types'], dataset['object_entity_idx'], dataset['object_entity_types'])):
        temp_sentence = dataset['sentence'].iloc[idx]
        temp_sentence = (temp_sentence[:e01_idx[0]] + f' @* ' + temp_sentence[e01_idx[0]:e01_idx[1]+1] + ' @ ' 
        + temp_sentence[e01_idx[1]+1:e02_idx[0]] + f' +^ ' + temp_sentence[e02_idx[0]:e02_idx[1]+1] + ' + ' + temp_sentence[e02_idx[1]+1:])
        dataset['sentence'].iloc[idx] = temp_sentence
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        #return_token_type_ids=False, # 문장 id
        )
  elif config.add_special_token == 'special':
    for idx, (e01_idx, e01_type, e02_idx, e02_type) in enumerate(zip(dataset['subject_entity_idx'], dataset['subject_entity_types'], dataset['object_entity_idx'], dataset['object_entity_types'])):
      temp_sentence = dataset['sentence'].iloc[idx]
      temp_sentence = (temp_sentence[:e01_idx[0]] + ' [E1] ' + temp_sentence[e01_idx[0]:e01_idx[1]+1] + ' [/E1] ' 
      + temp_sentence[e01_idx[1]+1:e02_idx[0]] + ' [E2] ' + temp_sentence[e02_idx[0]:e02_idx[1]+1] + ' [/E2] ' + temp_sentence[e02_idx[1]+1:])
      dataset['sentence'].iloc[idx] = temp_sentence
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        #return_token_type_ids=False, # 문장 id
        )
  else:
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False, # 문장 id
        )
  return tokenized_sentences

