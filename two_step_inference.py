import sys, getopt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
from utills import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
from tqdm import tqdm

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

def load_test_dataset(config, dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  config.prediction_mode = 'test'
  test_dataset = load_data(dataset_dir, config)
  test_label = list(map(int,test_dataset['label'].values))
  
  # tokenizing dataset
  tokenized_test = tokenized_dataset(config, test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def predict_output(model, dataset, device, mode):
  pred_answer, output_prob = inference(model, dataset, device) # model에서 class 추론
  if mode == 'multi':
    pred_answer = np.array(pred_answer) + 1
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  return pred_answer, output_prob

def change_probs(df, mode):
  print(list(df))
  new_first = {'id':[], 'pred_label':[], 'probs':[]}
  temp_p = np.zeros(30)
  if mode == 'binary':
    temp_p[0] = 1 # no_relation 확률을 1로
  
  for id_, pred_label, prob in df.values:
    if mode == 'multi':
      temp_p[1:] = prob[:]
    new_first['id'].append(id_)
    new_first['pred_label'].append(pred_label)
    new_first['probs'].append(list(temp_p))

  return pd.DataFrame(new_first)

def do_inference(config):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # 1. load tokenizer
  Tokenizer_NAME = config.model_name
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  if config.add_special_token:
    print("Add special token")
    # 추가 하고 싶은 Special token dict 정의
    special_tokens_dict = {'additional_special_tokens': config.new_special_token_list}
    # tokenizer에 더해주기
    tokenizer.add_special_tokens(special_tokens_dict)

  # 2. load my model
  if config.binary_model_path == 'None':
    MODEL_NAME = config.model_save_path # model dir.
    model1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model1.to(device)
  else:
    model1 = AutoModelForSequenceClassification.from_pretrained(config.binary_model_path)
    model2 = AutoModelForSequenceClassification.from_pretrained(config.multi_model_path)
    model1.to(device)
    model2.to(device)
  
  if config.add_special_token:
    model1.resize_token_embeddings(len(tokenizer))
    if config.binary_model_path != 'None':
      model2.resize_token_embeddings(len(tokenizer))
  
  # 3. First, inference with binary classification model
  ## load test datset
  test_dataset_dir = config.test_data_path
  test_id, test_dataset, test_label = load_test_dataset(config, test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)
  print("Data Amount:\t", len(test_id))
  ## predict answer
  pred_answer, output_prob = predict_output(model1, Re_test_dataset, device, 'binary')

  ## binary classification model ouput save
  first_output = pd.DataFrame({'id':test_id, 'indexs': np.arange(len(test_id)),
      'pred_label': pred_answer,'probs': output_prob, 'test_label': test_label})
  
  # 4. Second, inference with multi classification model
    # no_relation이라고 판단된 데이터를 제외하고 데이터셋 다시 구축
  Second_input = first_output.loc[first_output.pred_label != 'no_relation']
  Second_data = {
    'input_ids': test_dataset['input_ids'][list(Second_input['indexs'])],
    'token_type_ids': test_dataset['token_type_ids'][list(Second_input['indexs'])], 
    'attention_mask': test_dataset['attention_mask'][list(Second_input['indexs'])]
  }

  Re_test_dataset = RE_Dataset(Second_data , list(Second_input['test_label']))

  ## predict answer
  pred_answer, output_prob = predict_output(model2, Re_test_dataset, device, 'multi')
    ## 학습시 0~28을 사용했기 때문에, +1을 해줘야한다.
    ## 1~29를 사용하면 crossentropy 계열 로스를 계산할때 음수가 나오거나 할 수 있어서 오류가 발생한다.

  ## multi classification model output save
  second_output = pd.DataFrame({'id':Second_input['id'], 'pred_label':pred_answer,'probs':output_prob})

  # 5. Merge binary model ans multi model results
  first_output = first_output.drop(['indexs', 'test_label'], axis = 1)
  first_output = first_output.loc[first_output.pred_label == 'no_relation']

    # Pandas에서 리스트를 다루기 어렵기 때문에 dict를 이용해 prob을 변환하는 절차가 필요하다.
  first_output = change_probs(first_output, 'binary')
  second_output = change_probs(second_output, 'multi')

  output = pd.concat([first_output, second_output], axis=0)
  output = output.sort_values('id')
  print()
  submission_file_name = config.submission_file_name
  output.to_csv('./prediction/' + submission_file_name, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  print(output)
  print('---- Finish! ----')

if __name__ == '__main__':
    argv = sys.argv
    file_name = argv[0] # 실행시키는 파일명
    config_path = ""   # config file 경로

    try:
        # 파일명 이후 부터 입력 받는 옵션
        # help, config_path
        opts, etc_args = getopt.getopt(argv[1:], "hc:", ["help", "config_path="])
    except getopt.GetoptError:
        # 잘못된 옵션을 입력하는 경우
        print(file_name, "-c <config_path>")
        sys.exit(2)
        
    # 입력된 옵션을 적절히 변수로 입력
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(file_name, "-c <config_path>")
            sys.exit(0)
        elif opt in ("-c", "--config_path"):
            config_path = arg

    # 입력이 필수적인 옵션 입력이 없으면 오류 메시지 출력
    if len(config_path) < 1:
        print(file_name, "-c <config_path> is madatory")
        sys.exit(2)

    config = read_config(config_path)
    do_inference(config)

