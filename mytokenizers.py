from transformers import AutoTokenizer

def get_tokenizer(tokenizer_name):
    if tokenizer_name in ['klue/bert-base', 'klue/roberta-base', 'cosmoquester/bart-ko-base', 'tunib/electra-ko-base', 'monologg/koelectra-base-v3-discriminator']:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    return tokenizer