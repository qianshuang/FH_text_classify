# -*- coding: utf-8 -*-

import numpy as np
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import datasets
import json

model_name = "xlm-roberta-base"
finetuned_model_path = "./model/finetunedM"

# 加载数据集
dataset = datasets.load_dataset('csv', data_files={'train': './data/train_data.csv', 'test': './data/test_data.csv'})
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# label设置
with open('./data/label_to_id.json', 'r') as f:
    label_to_id = json.load(f)
id_to_label = {v: k for k, v in label_to_id.items()}

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_id), label2id=label_to_id, id2label=id_to_label)
metric = load_metric("./metrics/accuracy.py")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 定义训练参数
training_args = TrainingArguments(
    output_dir='./model',
    evaluation_strategy='steps',
    eval_steps=5,
    save_total_limit=2,
    num_train_epochs=3,
    save_steps=10,
    load_best_model_at_end=True
)

# 模型训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)
trainer.train()
trainer.evaluate()  # 直接跑传入的验证集

# 保存模型
tokenizer.save_pretrained(finetuned_model_path)
trainer.save_model(finetuned_model_path)
print("train finished...")
