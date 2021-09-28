import pandas as pd
import torch
from ast import literal_eval

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entities = []
  subject_idx = []
  object_entities = []
  object_idx = []
  for subject_entity, object_entity in zip(dataset['subject_entity'], dataset['object_entity']):
    subject_entity = literal_eval(subject_entity)
    object_entity = literal_eval(object_entity)

    subject_word = subject_entity['word']
    object_word = object_entity['word']

    subject_s_idx = int(subject_entity['start_idx'])
    subject_e_idx = int(subject_entity['end_idx'])

    object_s_idx = int(object_entity['start_idx'])
    object_e_idx = int(object_entity['end_idx'])
    
    subject_entities.append(subject_word)
    subject_idx.append([subject_s_idx, subject_e_idx])
    object_entities.append(object_word)
    object_idx.append([object_s_idx, object_e_idx])

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entities, 'subject_entity_idx': subject_idx,
          'object_entity':object_entities, 'object_entity_idx': object_idx,'label':dataset['label'],})

  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)

  return dataset

def tokenized_dataset(config, dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  if config.add_special_token:
    for idx, (e01_idx, e02_idx) in enumerate(zip(dataset['subject_entity_idx'], dataset['object_entity_idx'])):
      temp_sentence = dataset['sentence'].iloc[idx]
      temp_sentence = (temp_sentence[:e01_idx[0]] + '[e1]' + temp_sentence[e01_idx[0]:e01_idx[1]+1] + '[/e1]' 
      + temp_sentence[e01_idx[1]+1:e02_idx[0]] + '[e2]' + temp_sentence[e02_idx[0]:e02_idx[1]+1] + '[/e2]' + temp_sentence[e02_idx[1]+1:])
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
        #return_token_type_ids=False, # 문장 id
        )
  return tokenized_sentences

