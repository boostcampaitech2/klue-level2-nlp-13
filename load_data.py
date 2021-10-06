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
  subject_word, subject_idx, subject_type = [], [], []
  object_word, object_idx, object_type = [], [], []

  for subject_entity, object_entity in zip(dataset['subject_entity'], dataset['object_entity']):
    if type(subject_entity) == str:
      subject_dict = eval(subject_entity)
      object_dict = eval(object_entity)
    else:
      subject_dict = subject_entity
      object_dict = object_entity
      dataset['id'] = dataset.guid
    
    subject_word.append(subject_dict['word'])
    subject_idx.append((subject_dict['start_idx'], subject_dict['end_idx']))
    subject_type.append(subject_dict['type'])
    object_word.append(object_dict['word'])
    object_idx.append((object_dict['start_idx'], object_dict['end_idx']))
    object_type.append(object_dict['type'])
  
  out_dataset = pd.DataFrame({
      'id': dataset['id'], 
      'sentence': dataset['sentence'],
      'subject_word': subject_word,
      'subject_idx': subject_idx,
      'subject_type': subject_type,
      'object_word': object_word,
      'object_idx': object_idx,
      'object_type': object_type,
      'label': dataset['label'],
      'source': dataset['source']
  })
  
  return out_dataset

# def preprocessing_dataset(dataset):
#   """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
#   subject_entities = []
#   subject_idx = []
#   object_entities = []
#   object_idx = []
#   for subject_entity, object_entity in zip(dataset['subject_entity'], dataset['object_entity']):
#     subject_entity = literal_eval(subject_entity)
#     object_entity = literal_eval(object_entity)

#     subject_word = subject_entity['word']
#     object_word = object_entity['word']

#     subject_s_idx = int(subject_entity['start_idx'])
#     subject_e_idx = int(subject_entity['end_idx'])

#     object_s_idx = int(object_entity['start_idx'])
#     object_e_idx = int(object_entity['end_idx'])
    
#     subject_entities.append(subject_word)
#     subject_idx.append([subject_s_idx, subject_e_idx])
#     object_entities.append(object_word)
#     object_idx.append([object_s_idx, object_e_idx])

#   out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entities, 'subject_entity_idx': subject_idx,
#           'object_entity':object_entities, 'object_entity_idx': object_idx,'label':dataset['label'],})

#   return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)

  return dataset

def tokenized_dataset(config, dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []

  if config.add_special_token:
    
    if config.tokenizer_method == 'TEM':
      # Typed entity marker
      # 이순신 장군은 조선 출신이다. -> [CLS]<S:PER>이순신</S:PER>장군은 [SEP]<O:LOC>조선</O:LOC>출신이다.

      sentence_list = []
      seperate_token = '[SEP]'
      for s_word, s_type, s_idx, o_word, o_type, o_idx, sentence in zip(dataset['subject_word'], dataset['subject_type'], dataset['subject_idx'],
                                                                        dataset['object_word'], dataset['object_type'], dataset['object_idx'], dataset['sentence']):
        # s_idx = eval(s_idx)
        # o_idx = eval(o_idx)
        if s_idx > o_idx:
          sentence_list.append(sentence[:o_idx[0]] + f'<O:{o_type}>' + o_word + f'</O:{o_type}>' + sentence[o_idx[1]+1:s_idx[0]] 
                            + seperate_token + f'<S:{s_type}>' + s_word + f'</S:{s_type}>' + sentence[s_idx[1]+1:])
        else:
          sentence_list.append(sentence[:s_idx[0]] + f'<S:{s_type}>' + s_word + f'</S:{s_type}>' + sentence[s_idx[1]+1:o_idx[0]] 
                            + seperate_token + f'<O:{o_type}>' + o_word + f'</O:{o_type}>' + sentence[o_idx[1]+1:])

      if 'roberta' in config.model_name:
        tokenized_sentences = tokenizer(
            sentence_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
            return_token_type_ids=False, # 문장 id
        )
      else:
        tokenized_sentences = tokenizer(
            sentence_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
            #return_token_type_ids=False, # 문장 id
        )
    elif config.tokenizer_method == 'TEMP':
      # Typed entity marker Puntuation
      # 이순신 장군은 조선 출신이다. -> ... @ * PER * 이순신 @ ... # ∧ LOC ∧ 조선 # ...
      sentence_list = []

      for s_word, s_type, s_idx, o_word, o_type, o_idx, sentence in zip(dataset['subject_word'], dataset['subject_type'], dataset['subject_idx'],
                                                                        dataset['object_word'], dataset['object_type'], dataset['object_idx'], dataset['sentence']):
        # s_idx = eval(s_idx)
        # o_idx = eval(o_idx)
        if s_idx > o_idx:
            sentence_list.append(sentence[:o_idx[0]] + f' # ∧ {o_type} ∧ {o_word} # ' + sentence[o_idx[1]+1:s_idx[0]]
                                + f' @ * {s_type} * {s_word} @ ' + sentence[s_idx[1]+1:])
        else:
            sentence_list.append(sentence[:s_idx[0]] + f' @ * {s_type} * {s_word} @ ' + sentence[s_idx[1]+1:o_idx[0]]
                                + f' # ∧ {o_type} ∧ {o_word} # ' + sentence[o_idx[1]+1:])

      if 'roberta' in config.model_name:
        tokenized_sentences = tokenizer(
            sentence_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
            return_token_type_ids=False, # 문장 id
        )
      else:
        tokenized_sentences = tokenizer(
            sentence_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
            #return_token_type_ids=False, # 문장 id
        )
    else:  
      for idx, (e01_idx, e02_idx) in enumerate(zip(dataset['subject_idx'], dataset['object_idx'])):
        temp_sentence = dataset['sentence'].iloc[idx]
        temp_sentence = (temp_sentence[:e01_idx[0]] + '[E1]' + temp_sentence[e01_idx[0]:e01_idx[1]+1] + '[/E1]' 
        + temp_sentence[e01_idx[1]+1:e02_idx[0]] + '[E2]' + temp_sentence[e02_idx[0]:e02_idx[1]+1] + '[/E2]' + temp_sentence[e02_idx[1]+1:])
        dataset['sentence'].iloc[idx] = temp_sentence
      
      if 'roberta' in config.model_name:
        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
            return_token_type_ids=False, # 문장 id
        )
      else:
        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
            #return_token_type_ids=False, # 문장 id
        )
  else:
      for e01, e02 in zip(dataset['subject_word'], dataset['object_word']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)

      if 'roberta' in config.model_name:
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
            return_token_type_ids=False, # 문장 id
        )
      else:
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
            #return_token_type_ids=False, # 문장 id
        )
  print('====== print tokenized sentence sample ======')
  print(tokenizer.decode(tokenized_sentences.input_ids[0]))
  return tokenized_sentences

