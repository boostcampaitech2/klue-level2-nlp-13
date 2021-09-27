from transformers import AutoConfig, AutoModelForSequenceClassification

def get_model(model_name, num_classes):
    if model_name in ['klue/bert-base', 'klue/roberta-base']:
        model_config =  AutoConfig.from_pretrained(model_name)
        model_config.num_labels = num_classes
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    return model


