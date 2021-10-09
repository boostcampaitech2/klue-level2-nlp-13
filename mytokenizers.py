##################
# import modules #
##################

from transformers import AutoTokenizer

#############
# functions #
#############

def get_tokenizer(tokenizer_name):
    """
        get available naming tokenizer
    """
    # ['klue/bert-base', 'klue/roberta-base', 'cosmoquester/bart-ko-base', 'tunib/electra-ko-base', 'monologg/koelectra-base-v3-discriminator']
    if  tokenizer_name == 'custom_robert_base':
        tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    elif tokenizer_name == 'custom_robert_large':
        tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    return tokenizer