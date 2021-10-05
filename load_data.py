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



def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)

  # str to dict
  pd_dataset["subject_entity"] = pd_dataset["subject_entity"].apply(lambda x : eval(x))
  pd_dataset["object_entity"] = pd_dataset["object_entity"].apply(lambda x : eval(x))
  
  return pd_dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01["word"] + '[SEP]' + e02["word"]
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


def typed_entity_marker_punct(pd_dataset, tokenizer, max_length):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  type_to_ko = {"PER":"사람", "ORG":"단체", "POH" : "기타", "LOC" : "장소", "NOH" : "수량", "DAT" : "날짜"} 

  sentence_list = []
  for subject_dict, object_dict, sentence in zip(pd_dataset['subject_entity'], pd_dataset['object_entity'], pd_dataset["sentence"]):
    #['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
    # ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
    # sentence : 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
    #  "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}"
    
    sub_start = subject_dict["start_idx"]
    sub_end = subject_dict["end_idx"] + 1

    obj_start = object_dict["start_idx"]
    obj_end = object_dict["end_idx"] + 1

    object_type = object_dict["type"]
    subject_type = subject_dict["type"]

    if subject_dict["end_idx"] < object_dict["end_idx"]:
      sentence = (sentence[:sub_start]
                  + ' @ * ' + type_to_ko[subject_type] + " * " + sentence[sub_start:sub_end] + " @ "
                  + sentence[sub_end:obj_start] 
                  + " # ^ " + type_to_ko[object_type] + " ^ " +sentence[obj_start:obj_end] + " # " 
                  + sentence[obj_end:]
                )
    else :
      sentence = (sentence[:obj_start]
                  + " # ^ " + type_to_ko[object_type] + " ^ " +sentence[obj_start:obj_end] + " # " 
                  + sentence[obj_end:sub_start] 
                  + ' @ * ' + type_to_ko[subject_type] + " * " + sentence[sub_start:sub_end] + " @ "
                  + sentence[sub_end:]
                )

    sentence_list.append(sentence)
  
  tokenized_sentences = tokenizer(
      sentence_list,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=max_length,
      add_special_tokens=True,
      return_token_type_ids = False,
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
