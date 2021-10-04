import pickle as pickle
import os
import pandas as pd
import torch


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

# def preprocessing_dataset(dataset):
#   """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
#   subject_entity = []
#   object_entity = []
#   for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
#     i = i[1:-1].split(',')[0].split(':')[1]
#     j = j[1:-1].split(',')[0].split(':')[1]

#     subject_entity.append(i)
#     object_entity.append(j)
#   out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
#   return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
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
      )
  return tokenized_sentences


import json
import pandas as pd
from collections import defaultdict

train_json_path = "/opt/ml/KLUE-Baseline/data/klue_benchmark/klue-re-v1.1/klue-re-v1.1_train.json"
dev_json_path = "/opt/ml/KLUE-Baseline/data/klue_benchmark/klue-re-v1.1/klue-re-v1.1_dev.json"

def json_to_df(json_path):
    with open(json_path) as f:
        json_object = json.load(f)
    data = defaultdict(list)
    for dict in json_object:
        for key, value in dict.items():
            data[key].append(value) 
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":

  def label_to_num(label : list) -> list:
    num_label = []
    with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
    for v in label:
      num_label.append(dict_label_to_num[v])
    
    return num_label

  from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer

  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = "klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("/opt/ml/dataset/train/train.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  train_label = label_to_num(train_dataset['label'].values)
  # dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)


#%%
print(tokenizer.decode(RE_train_dataset[0]["input_ids"]))
