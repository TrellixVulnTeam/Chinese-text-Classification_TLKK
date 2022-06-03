import os
import pandas as pd
from sklearn.model_selection import train_test_split

xlsx_path = r"F:\Bert-Chinese-Text-Classification-Pytorch-master\data\train.csv"
xlsx_path_test = r"F:\Bert-Chinese-Text-Classification-Pytorch-master\data\testA.csv"
train_path = r"F:\Bert-Chinese-Text-Classification-Pytorch-master\data\train.txt"
val_path = r"F:\Bert-Chinese-Text-Classification-Pytorch-master\data\dev.txt"
test_path = r"F:\Bert-Chinese-Text-Classification-Pytorch-master\data\test.txt"


train = pd.read_csv(xlsx_path)

x_train, x_test, train_label, test_label = train_test_split(
    train['text'].tolist(), train['label'].tolist(), test_size=0.2, stratify=train['label'].tolist())

test = pd.read_csv(xlsx_path_test)

test['label'] = 0#设置测试集标签为0
x_testA = test['text'].tolist()
x_testA_label = test['label'].tolist()

with open(train_path, 'a', encoding='utf-8') as f:
    for index, text in enumerate(x_train):
        label = str(train_label[index])
        if '\t' in text:
            text = text.replace('\t', '')
        content = text + '\t' + label + '\n'
        f.write(content)

with open(val_path, 'a', encoding='utf-8') as g:
    for index, text in enumerate(x_test):
        label = str(test_label[index])
        if '\t' in text:
            text = text.replace('\t', '')
        content = text + '\t' + label + '\n'
        g.write(content)

with open(test_path, 'a', encoding='utf-8') as h:
    for index, text in enumerate(x_testA):
        label = str(x_testA_label[index])
        if '\t' in text:
            text = text.replace('\t', '')
        content = text + '\t' + label + '\n'
        h.write(content)