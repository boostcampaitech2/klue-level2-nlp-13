from transformers import AutoConfig, AutoModelForSequenceClassification
#from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
#from transformers import RobertaConfig, RobertaForSequenceClassification

def get_model(model_name, num_classes):
    if model_name == 'klue/bert-base':
        model_config =  AutoConfig.from_pretrained(model_name)
        model_config.num_labels = num_classes
        model =  AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    return model


