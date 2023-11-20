# -*- coding: utf-8 -*-

import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

finetuned_model_path = "./model/finetunedM"

with open('./data/label_to_id.json', 'r') as f:
    label_to_id = json.load(f)
id_to_label = {v: k for k, v in label_to_id.items()}

#
# 模型测试
finetunedM = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path, device_map={"": "cuda:1"})
tokenizerM = AutoTokenizer.from_pretrained(finetuned_model_path)

sequences = ["张三的电话是多少啊？", "SSE的责任人是谁？"]
tokens = tokenizerM(sequences, padding="max_length", truncation=True, return_tensors="pt")
outputs = finetunedM(**tokens)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
scores, predicted_labels = torch.max(predictions, dim=-1)
print(scores.tolist(), [id_to_label[i] for i in predicted_labels.tolist()])
