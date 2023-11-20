# -*- coding: utf-8 -*-

import random
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import *

# 加载数据
data = []
columns_to_convert = ['知识标题', '相似问法', '答案内容（网页）']
for file_name in ['./data/SSE-2023-07-12-10_45_54.xlsx', './data/责任人-07-12-10_55_04.xlsx', './data/Expense-Receipt.xlsx']:
    df = pd.read_excel(file_name, engine='openpyxl')
    df[columns_to_convert] = df[columns_to_convert].astype(str)
    df.drop_duplicates(inplace=True)

    for index, row in df.iterrows():
        data.append(row['知识标题'].strip())

data_nomal = random.sample(data, 100)
data_privacy = read_lines("data/privacy.txt")

# 构造训练数据
df_nomal = pd.DataFrame({'text': data_nomal, 'label': 'nomal'})
df_privacy = pd.DataFrame({'text': data_privacy, 'label': 'privacy'})
df = pd.concat([df_nomal, df_privacy], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
print(df)

# label设置
unique_labels = df['label'].unique()
label_to_id = {label: i for i, label in enumerate(unique_labels)}
df['label'] = df['label'].replace(label_to_id)
write_json_2_file(label_to_id, './data/label_to_id.json')
print(df)

# 数据集拆分并存储
df_train_data, df_test_data = train_test_split(df, test_size=0.1, random_state=42)
print(f'Train: {len(df_train_data)}, test: {len(df_test_data)}')
df_train_data.to_csv("./data/train_data.csv", encoding="utf-8", index=False)
df_test_data.to_csv("./data/test_data.csv", encoding="utf-8", index=False)
