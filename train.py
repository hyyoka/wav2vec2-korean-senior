from datasets import load_metric, load_from_disk
# from data_proc import  get_senior_data
from data_collator import DataCollatorCTCWithPadding
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import torch
import ast

import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

repo_name = 'wav2vec2-xlsr-korean-senior'

processor = Wav2Vec2Processor.from_pretrained("hyyoka/wav2vec2-xlsr-korean-senior")
wer_metric = load_metric("wer")
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = 49 # as fleek model has 2 pad tokens
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def get_model(model_name="hyyoka/wav2vec2-xlsr-korean-senior"): 
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name, 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean", 
        pad_token_id=49,
        vocab_size=50,
        ignore_mismatched_sizes=True
    )

    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()
    return model

def train_model(train, test, model):
    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=300,
        eval_steps=300,
        logging_steps=50,
        learning_rate=3e-4,
        warmup_steps=300,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        dataloader_num_workers=6
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator
    )
    print(f"build trainer on device {training_args.device} with {training_args.n_gpu} gpus")
    trainer.train()

if __name__ == "__main__":
    
    # torch.cuda.empty_cache()
    
    dataset = load_from_disk('./elders_dataset')
    dataset = dataset.remove_columns(['wav', 'text', 'labels'])

    train = dataset['train'] #.select([i for i in range(0,5000)])
    test =  dataset['valid'] #.select([i for i in range(0,5000)])

    model = get_model()
    train_model(train, test, model)
    

    