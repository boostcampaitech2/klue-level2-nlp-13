from transformers import AutoTokenizer

# custom
from tokenizers import BertWordPieceTokenizer
from Korpora import Korpora
import json

def get_tokenizer(tokenizer_name):
    if tokenizer_name in ['klue/bert-base', 'klue/roberta-base','klue/roberta-large','xlm-roberta-large','ainize/klue-bert-base-re','google/mt5-large']:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # else:
    #     corpus = Korpora.load("kcbert")
    #     limit_alphabet = 22000
    #     vocab_size = 10000

    #     tokenizer = BertWordPieceTokenizer(
    #         vocab_file = None,
    #         clean_text = True,
    #         handle_chinese_chars = True,
    #         strip_accents = False,
    #         lowercase = False,
    #         wordpeices_prefix = "##"
    #     )

    #     tokenizer.train(
    #         files = corpus,
    #         limit_alphabet = limit_alphabet, # default
    #         vocab_size = vocab_size # default
    #     )
    #     tokenizer.save("./vocab/ch-{}-wpm-{}-pretty".format(limit_alphabet, vocab_size),True)

    return tokenizer