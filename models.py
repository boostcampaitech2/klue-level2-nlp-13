from transformers import AutoConfig, AutoModelForSequenceClassification, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from torchsummary import summary

class MyRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size*3, config.hidden_size*2)
        self.dense2 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, entity_location):
        #entity_location [subj_s, subj_e, obj_s, obj_e]
        #print(entity_location)
        subject_entity_avgs = []
        oject_entity_avgs = []
        for idx in range(entity_location.shape[0]):
            subject_entity_avg = features[:, entity_location[idx][0]:entity_location[idx][1], :]
            subject_entity_avg = torch.mean(subject_entity_avg, axis=1)
            subject_entity_avgs.append(subject_entity_avg.cpu().detach().numpy())

            oject_entity_avg = features[:, entity_location[idx][2]:entity_location[idx][3], :]
            oject_entity_avg = torch.mean(oject_entity_avg, axis=1)
            oject_entity_avgs.append(oject_entity_avg.cpu().detach().numpy())
        
       # print(subject_entity_avgs)

        subject_entity_avgs = torch.tensor(subject_entity_avgs)
        oject_entity_avgs = torch.tensor(oject_entity_avgs)
        #print(subject_entity_avg.shape, oject_entity_avg.shape, features[:, 0, :].shape)
        
        x = torch.cat([features[:, 0, :], subject_entity_avg, oject_entity_avg], axis = 1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        #x = self.dropout(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class MyRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classification = MyRobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        entity_location=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            entity_type_ids=entity_type_ids
        )
        sequence_output = outputs[0]     
        logits = self.classification(sequence_output, entity_location)

        loss = None
        if labels is not None:
           loss = 0

        #if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

def get_model(model_name, num_classes):
    if  model_name == 'custom_robert_base':
        model_config =  AutoConfig.from_pretrained('./roberta-retrained/base/')
        model_config.num_labels = num_classes
        model = MyRobertaForSequenceClassification.from_pretrained('./roberta-retrained/base/', config=model_config)#'klue/roberta-base', config=model_config)
    elif  model_name == 'custom_robert_large':
        model_config =  AutoConfig.from_pretrained('./roberta-retrained/large/')
        model_config.num_labels = num_classes
        model = MyRobertaForSequenceClassification.from_pretrained('./roberta-retrained/large/', config=model_config)#'klue/roberta-base', config=model_config)
    else:
        model_config =  AutoConfig.from_pretrained(model_name)
        model_config.num_labels = num_classes
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    return model