#%%
import pandas as pd
import torch
import sklearn
import numpy as np
from load_data import *
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

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

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = json_to_df(dataset_dir)
  
  test_label = list(map(int, label_to_num(test_dataset['label'].values)))
  
  # tokenizing dataset
  tokenized_test, _ = entity_marker_punct(test_dataset, tokenizer, 200, argu = False, state ="inference")

  return tokenized_test, test_label

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size = 64, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          # token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


# if __name__ == '__main__':

# parser = argparse.ArgumentParser()
# parser.add_argument('--model_dir', type=str, default="/opt/ml/klue-level2-nlp-13/best_model/roberta-large")
# args = parser.parse_args()

"""
  주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# load tokenizer
Tokenizer_NAME = "klue/roberta-large" # args.tokenizer_name
tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

## load my model
MODEL_NAME = "/opt/ml/klue-level2-nlp-13/best_model/roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)

## load test datset

dev_json_path = "/opt/ml/KLUE-Baseline/data/klue_benchmark/klue-re-v1.1/klue-re-v1.1_dev.json"
test_dataset, test_label = load_test_dataset(dev_json_path, tokenizer)

Re_test_dataset = RE_Dataset(test_dataset, test_label)

## predict answer
pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론

#%%
output_prob


#%%
df = json_to_df(dev_json_path)
test_label = label_to_num(df["label"].values)
#%%
pred_answer


#%%
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

print(klue_re_micro_f1(pred_answer, test_label))
# print(klue_re_micro_f1(output_prob, output_prob))
    #, labels=label_indices
# %%
from sklearn.metrics import f1_score
y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]

    #, labels=label_indices
    #, labels=label_indices
f1_score(output_prob, output_prob, average="micro")

# f1_score(y_true, y_pred, average="micro")

#%%
len(output_prob[1])