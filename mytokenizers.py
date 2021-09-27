from transformers import AutoTokenizer
#from transformers import RobertaTokenizer, BertTokenizer

def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'klue/bert-base':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    return tokenizer