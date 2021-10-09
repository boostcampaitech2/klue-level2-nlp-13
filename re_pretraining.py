##################
# import modules #
##################

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

#################
# set variables #
#################

os.environ["TOKENIZERS_PARALLELISM"] = "false"


########
# main #
########

if __name__ == "__main__":
    """
        Line by Line으로 data를 읽어 MLM task로 RoBERTa 모델 pre-train을 수행합니다.
    """
     # set tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    model = AutoModelForMaskedLM.from_pretrained('klue/roberta-large')


     # load data and set dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="../../dataset/train/train.csv",
        block_size=512,
    )

     # set mlm task
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

     # set training args
    training_args = TrainingArguments(
        output_dir="./roberta-retrained",
        overwrite_output_dir=True,
        num_train_epochs=15,
        per_device_train_batch_size=16,
        evaluation_strategy = 'steps',
        save_steps=500,
        save_total_limit=5,
        load_best_model_at_end=True,
        seed=42,
    )
     # set Trainer class for pre-training
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset,    
    )

     # Start traing
    trainer.train()
    
     # Save result
    trainer.save_model("./roberta-retrained_model")