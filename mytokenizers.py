from transformers import AutoTokenizer

def get_tokenizer(tokenizer_name):
    # ['klue/bert-base', 'klue/roberta-base', 'cosmoquester/bart-ko-base', 'tunib/electra-ko-base', 'monologg/koelectra-base-v3-discriminator']
    if  tokenizer_name == 'custom_robert':
        tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    return tokenizer