#%%
import pickle as pickle
import pandas as pd
import torch
import random
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
    for key in df["subject_entity"][0].keys():
      df["subject_" + key] =  df["subject_entity"].apply(lambda x : x[key])
      df["object_" + key] =  df["object_entity"].apply(lambda x : x[key])  

    return df

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)

  # str to dict
  pd_dataset["subject_entity"] = pd_dataset["subject_entity"].apply(lambda x : eval(x))
  pd_dataset["object_entity"] = pd_dataset["object_entity"].apply(lambda x : eval(x))

  for key in pd_dataset["subject_entity"][0].keys():
    pd_dataset["subject_" + key] =  pd_dataset["subject_entity"].apply(lambda x : x[key])
    pd_dataset["object_" + key] =  pd_dataset["object_entity"].apply(lambda x : x[key])  
  return pd_dataset


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, tokenized_tensor, labels):
    self.tokenized_tensor = tokenized_tensor
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.tokenized_tensor.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


def tokenized_dataset(pd_dataset, tokenizer, max_length):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(pd_dataset['subject_entity'], pd_dataset['object_entity']):
    temp = ''
    temp = e01["word"] + '[SEP]' + e02["word"]
    concat_entity.append(temp)
  
  tokenized_sentences = tokenizer(
      concat_entity,
      list(pd_dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=max_length,
      add_special_tokens=True,
      )
  return tokenized_sentences, pd_dataset['label'].values


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

  return tokenized_sentences, pd_dataset['label'].values

def mix_word(pd_dataset, except_label = []):
  # 라벨별로 서로 단어들을 섞어준다.
  with open('dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
  labels = dict_label_to_num.keys()
  labels = list(labels)

  # 섞는 것을 제한하는 라벨 설정
  for label in except_label:
    labels.remove(label) #remove는 오로지 value만 가능.

  for label in labels:
    # sub, obj 나눠서 해야함.
    for relation in ["subject_word", "object_word"]:
      label_words = pd_dataset.query("label == @label")[relation]
      index = label_words.index
      words = list(label_words)
      # 단어 섞어서 다시 재배치
      random.shuffle(words) #print 할경우 None 반환
      pd_dataset.loc[index, relation] = words
  return pd_dataset


def argue_dataset(df):
    
    # argu data num calculate
    label_count = df["label"].value_counts()
    #나눠 주는 값은 전체 데이터셋에서 가장 큰 비중인 no_relation의 비율
    result = label_count/len(df["label"])/0.293625
    #그냥 하면 너무 크기때문에 3을 나눠줌
    label_count_dict = dict(((1/result)/3) * label_count - label_count )

    #증강 안해도 많은 양을 가지고 있는 라벨은 그대로 둔다.
    del label_count_dict["no_relation"]
    del label_count_dict['org:top_members/employees']
    del label_count_dict['per:employee_of']

    argu_df = pd.DataFrame()
    for label, count in label_count_dict.items():
        upsample_df = df.query("label == @label").sample(n = int(count), replace = True)
        argu_df = pd.concat([argu_df, upsample_df])
        
    argu_df = argu_df.sample(frac=1).reset_index(drop=True)
    argu_df = mix_word(argu_df)

    return argu_df

def argu2(pd_dataset):
  argu_df = mix_word(pd_dataset, ["no_relation"])

  for label in ["no_relation", 'org:top_members/employees', 'per:employee_of']:
    argu_df = argu_df.drop(argu_df.query("label == @label").index)
  
  return argu_df

def entity_marker_punct(pd_dataset, tokenizer, max_length = 256, argu = False):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""

  # if argu:
  #   argu_df = argue_dataset(pd_dataset)
  argu_df = argu2(pd_dataset)
  pd_dataset = pd.concat([pd_dataset, argu_df ])

  sentence_list = []
  for subject_dict, object_dict, sentence, sub_word, obj_word in zip(pd_dataset['subject_entity']
                                                                    , pd_dataset['object_entity']
                                                                    , pd_dataset["sentence"]
                                                                    , pd_dataset["subject_word"]
                                                                    , pd_dataset["object_word"]
                                                                    ):
    #['@'] + tokens_wordpiece + "@"
    # ["#"] + tokens_wordpiece + "#"
    # sentence : 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
    #  "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}"
    
    sub_start = subject_dict["start_idx"]
    sub_end = subject_dict["end_idx"] + 1

    obj_start = object_dict["start_idx"]
    obj_end = object_dict["end_idx"] + 1

    if subject_dict["end_idx"] < object_dict["end_idx"]:
      sentence = (sentence[:sub_start]
                  + ' @ '  + sub_word + " @ "
                  + sentence[sub_end:obj_start] 
                  + " # "  + obj_word + " # " 
                  + sentence[obj_end:]
                )
    else :
      sentence = (sentence[:obj_start]
                  + " #  "  + obj_word + " # " 
                  + sentence[obj_end:sub_start] 
                  + ' @  ' + sub_word + " @ "
                  + sentence[sub_end:]
                )

    sentence_list.append(sentence)
  
  tokenized_sentences = tokenizer(
      sentence_list,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length = max_length,
      add_special_tokens=True,
      return_token_type_ids = False,
      )

  def label_to_num(label : list) -> list:
    num_label = []
    with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
    for v in label:
      num_label.append(dict_label_to_num[v])
    return num_label

  labels = label_to_num(pd_dataset['label'].values)

  return tokenized_sentences, labels


if __name__ == "__main__":

  def label_to_num(label : list) -> list:
    num_label = []
    with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
    for v in label:
      num_label.append(dict_label_to_num[v])
    return num_label

  from transformers import (AutoTokenizer, )

  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = "klue/roberta-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("/opt/ml/dataset/train/train.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  # tokenizing dataset
  # tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # # make tokenized tensor, typed_entity_marker_punct
  # tokenized_train = typed_entity_marker_punct(train_dataset, tokenizer)
  # # tokenized_dev = typed_entity_marker_punct(dev_dataset, tokenizer)

  # make tokenized tensor, typed_entity_marker_punct
  tokenized_train, train_label = entity_marker_punct(train_dataset, tokenizer, argu = True)
  # tokenized_dev, dev_label = entity_marker_punct(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)


# #%%
# len(tokenized_train["input_ids"])
# #%%
# len(RE_train_dataset)
# # #%%
# # print(tokenizer.decode(RE_train_dataset[3]["input_ids"]))
# # # %%
