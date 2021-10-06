#%%
# klue-dev와 ai stage test의 비교 << 동일
import pandas as pd
from transformers import AutoTokenizer, AutoConfig,AutoModelForMaskedLM, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer


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

dev_df = json_to_df(dev_json_path)

dev_df["sentence_len"] = dev_df["sentence"].apply(lambda x : len(x))


df = pd.read_csv("/opt/ml/dataset/test/test_data.csv")
df["sentence_len"] = df["sentence"].apply(lambda x : len(x))

#%%
df["sentence_len"].value_counts()
#%%
dev_df["sentence_len"].value_counts()

test_0 = "지난 15일 MBC '탐사기획 스트레이트'가 이 사실을 보도했다."

dev_df[dev_df["sentence"] == test_0]