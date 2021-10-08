# Re pretraining for roberta-large model
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
from transformers import LineByLineTextDataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    model = AutoModelForMaskedLM.from_pretrained('klue/roberta-large')

    #train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset['label'], shuffle=True, random_state=42)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="../../dataset/train/train.csv",
        block_size=512,
    )   

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset,    
    )

    trainer.train()
    trainer.save_model("./roberta-retrained_model")