#%%
import pandas as pd
from transformers import AutoTokenizer, AutoConfig,AutoModelForMaskedLM, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer

#csv로드 후 str to dict 해준다.
df = pd.read_csv("/opt/ml/dataset/train/train.csv")
df["subject_entity"] = df["subject_entity"].apply(lambda x : eval(x))
df["object_entity"] = df["object_entity"].apply(lambda x : eval(x))

for key in df["subject_entity"][0].keys():
    print(key)
    df["subject_" + key] =  df["subject_entity"].apply(lambda x : x[key])
    df["object_" + key] =  df["object_entity"].apply(lambda x : x[key])

#%%

# label별로 sub와 obj의 type을 탐색.
import pickle

with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)

labels = dict_label_to_num.keys()

for label in labels:
    print(label, "object_type")
    print(df[df["label"] == label]["object_type"].value_counts())
    print(label, "subject_type")
    print(df[df["label"] == label]["subject_type"].value_counts())
    print("=" * 30)


#%%
# "per:place_of_death"의 데이터를 몇가지 탐색
df[df["label"] == "per:place_of_death"]["subject_type"].value_counts()
df.query('label == "per:place_of_death" and object_type == "DAT"')
df.query('label == "per:place_of_death" and object_type == "PER"')

#%%
import pandas as pd
from transformers import AutoTokenizer, AutoConfig,AutoModelForMaskedLM, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer

df = pd.read_csv("/opt/ml/dataset/train/train.csv")

MODEL_NAME = "klue/bert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#%%
df.apply(lambda x : len(x["sentence"]))

df["sentence_token"] =df["sentence"].apply(lambda x :tokenizer.encode(x))

#%%
df["sentence_token_len"] = df["sentence_token"].apply(lambda x :len(x))
#%%
df["sentence_token_len"].plot()

#%%
df["subject_end"] = df["subject_entity"].apply(lambda x : x["end_idx"])
df["object_end"]= df["object_entity"].apply(lambda x : x["end_idx"])

df["sentence_token_len_limit"] = df.apply(lambda x : tokenizer.encode(x["sentence"][:max(x["subject_end"], x["object_end"]) + 1]))
df["sentence_token_len_limit"].plot()
# df["last"] = df[["subject_end", "object_end"]].max(axis =1 )
# df["last"].plot()
#%%

df["sentence_token"] =df["sentence"].apply(lambda x :tokenizer.encode(x))
df["sentence_token_len"] = df["sentence_token"].apply(lambda x :len(x))
df["sentence_token_len"].plot()
