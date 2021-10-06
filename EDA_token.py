
#%%
MODEL_NAME = "klue/bert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ids = tokenizer.convert_tokens_to_ids(["사람", "단체", "기타", "장소", "수량", "날짜"])

# ids  = tokenizer.convert_tokens_to_ids([x.lower() for x in df["object_entity_type"].value_counts().index])
print(ids)
tokens = tokenizer.convert_ids_to_tokens(ids)
print(tokens)
# %%
# %%
ids = tokenizer.encode(" @ ^ * # ")
print(ids)
tokens = tokenizer.decode(ids)
print(tokens)
# %%
"df " + "df"
# %%

#%%
sent = "abc"

sent[:2] + "@" + sent[2:]
# %%

type_to_ko = {"PER":"사람", "ORG":"단체", "POH" : "기타", "LOC" : "장소", "NOH" : "수량", "DAT" : "날짜"} 

sentence = "〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다."
subject_dict = {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}
object_dict = {'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}

    
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
                + " # ^ " + type_to_ko[object_type] + " ^ " +sentence[obj_start:obj_end ] + " # " 
                + sentence[obj_end:sub_start] 
                + ' @ * ' + type_to_ko[subject_type] + " * " + sentence[sub_start :sub_end ] + " @ "
                + sentence[sub_end:]
            )
print(sentence)
# %%