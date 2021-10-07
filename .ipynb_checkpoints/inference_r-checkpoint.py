from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader

from load_data import load_data, RE_Dataset
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    type_to_ko = {"PER":"사람", "ORG":"단체", "POH" : "기타", "LOC" : "장소", "NOH" : "수량", "DAT" : "날짜"} 
    sentence_list = []

    for subject_dict, object_dict, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset["sentence"]):
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
        max_length=200,
        add_special_tokens=True,
        return_token_type_ids = False,
        )

    return tokenized_sentences


def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device)
          # token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

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

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  ## load my model
  MODEL_NAME = './results/checkpoint-30000' # model dir.
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission_siaun_30000.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default='./results/checkpoint-30000')
  args = parser.parse_args()
  print(args)
  main(args)
  
