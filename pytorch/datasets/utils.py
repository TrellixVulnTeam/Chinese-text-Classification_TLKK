import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def split_train_val(config):
    train = pd.read_csv(config.train_data)
    # 划分为训练集和验证集
    # stratify 按照标签进行采样，训练集和验证部分同分布
    x_train, x_val, train_label, val_label = train_test_split(
        train['text'].tolist(), train['label'].tolist(), test_size=0.2, stratify=train['label'].tolist())

    test = pd.read_csv(config.test_data)
    test['label'] = 0  # 设置测试集标签为0
    x_testA = test['text'].tolist()
    x_testA_label = test['label'].tolist()

    return x_train, train_label, x_val, val_label, x_testA, x_testA_label

def text_encoding(x_train, x_val, x_testA):

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
    val_encoding = tokenizer(x_val, truncation=True, padding=True, max_length=64)
    testA_encoding = tokenizer(x_testA, truncation=True, padding=True, max_length=64)

    return train_encoding, val_encoding, testA_encoding

class NewsDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


