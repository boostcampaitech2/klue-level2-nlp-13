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

        subject_type = subject_entity["type"]
        object_type = object_entity["type"]

        subject_entities.append(subject_word)
        subject_idx.append([subject_s_idx, subject_e_idx])
        object_entities.append(object_word)
        object_idx.append([object_s_idx, object_e_idx])

    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,
        'object_entity':object_entity,'subject_type':subject_type,'object_type':object_type,'label':dataset['label']})

    return out_dataset

# def load_data(dataset_dir):
#     """ csv 파일을 경로에 맡게 불러 옵니다. """
#     pd_dataset = pd.read_csv(dataset_dir)
#     dataset = preprocessing_dataset(pd_dataset)

#     return dataset


def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)

  # str to dict
  pd_dataset["subject_entity"] = pd_dataset["subject_entity"].apply(lambda x : eval(x))
  pd_dataset["object_entity"] = pd_dataset["object_entity"].apply(lambda x : eval(x))
  
  return pd_dataset
  
def tokenized_dataset(config, dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    type_to_ko = {"PER":"사람", "ORG":"단체", "POH" : "기타", "LOC" : "장소", "NOH" : "수량", "DAT" : "날짜"} 
    sentence_list = []

    for subject_dict, object_dict, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset["sentence"]):
        #['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
        # ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
        # sentence : 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.
        #  "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}"
        print(subject_dict)
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

    
    # if config.add_special_token:
    #     for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    #         temp = ''
    #         temp = e01 + '[SEP]' + e02
    #         concat_entity.append(temp)
    #     for idx, (e01_idx, e02_idx,subj_type,obj_type) in enumerate(zip(dataset['subject_entity_idx'], dataset['object_entity_idx'],dataset['subject_type'],dataset['object_type'])):
    #         temp_sentence = dataset['sentence'].iloc[idx]
    #         temp_sentence = (temp_sentence[:e01_idx[0]] + '<subj>' + type_to_ko[subj_type] + '</subj>' 
    #         + temp_sentence[e01_idx[1]+1:e02_idx[0]] + '<obj>' + type_to_ko[obj_type] + '<obj>' + temp_sentence[e02_idx[1]+1:])
    #         dataset['sentence'].iloc[idx] = temp_sentence
            

    #     # for idx, (e01_idx, e02_idx) in enumerate(zip(dataset['subject_entity_idx'], dataset['object_entity_idx'])):
    #     #     temp_sentence = dataset['sentence'].iloc[idx]
    #     #     # temp_sentence = dataset['sentence'][idx]
    #     #     temp_sentence = (temp_sentence[:e01_idx[0]] + '<subj>' + temp_sentence[e01_idx[0]:e01_idx[1]+1] + '</subj>' 
    #     #     + temp_sentence[e01_idx[1]+1:e02_idx[0]] + '<obj>' + temp_sentence[e02_idx[0]:e02_idx[1]+1] + '<obj>' + temp_sentence[e02_idx[1]+1:])
    #     #     dataset['sentence'].iloc[idx] = temp_sentence
    #     #     # dataset['sentence'][idx] = temp_sentence
    #     tokenized_sentences = tokenizer(
    #         # concat_entity,
    #         list(dataset['sentence']),
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=200,
    #         add_special_tokens=True,
    #         # return_token_type_ids=False, # 문장 id
    #         )
    # else:
    #     for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    #         temp = ''
    #         temp = e01 + '[SEP]' + e02
    #         concat_entity.append(temp)
    #     for idx, (e01_idx, e02_idx,subj_type,obj_type) in enumerate(zip(dataset['subject_entity_idx'], dataset['object_entity_idx'],dataset['subject_type'],dataset['object_type'])):
    #         temp_sentence = dataset['sentence'].iloc[idx]
    #         # temp_sentence = dataset['sentence'][idx]
    #         temp_sentence = (temp_sentence[:e01_idx[0]] + '<subj>' + type_to_ko[subj_type] + '</subj>' 
    #         + temp_sentence[e01_idx[1]+1:e02_idx[0]] + '<obj>' + type_to_ko[obj_type] + '<obj>' + temp_sentence[e02_idx[1]+1:])
    #         dataset['sentence'].iloc[idx] = temp_sentence
    #         # dataset['sentence'][idx] = temp_sentence
    #     tokenized_sentences = tokenizer(
    #         concat_entity,
    #         list(dataset['sentence']),
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=128,
    #         add_special_tokens=True,
    #         # return_token_type_ids=False, # 문장 id
    #         )
    # return tokenized_sentences
