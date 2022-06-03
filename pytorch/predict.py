import torch
import numpy as np
from datasets.utils import split_train_val, text_encoding, NewsDataset
from torch.utils.data import DataLoader
import pandas as pd
import datetime
from models.bert import Bert_Pytorch
from config import Config

def predict(model, config, device):
    model.load_state_dict(torch.load(config.save_dict.format(9), map_location='cpu'))
    model.eval()

    test = pd.read_csv(config.test_data)
    x_train, _, x_val, _, x_testA, x_testA_label = split_train_val(config)
    _, _, testA_encoding = text_encoding(x_train, x_val, x_testA)
    testA_dataset = NewsDataset(testA_encoding, x_testA_label)

    testA_dataloader = DataLoader(testA_dataset, batch_size=config.batch_size, shuffle=False)

    print('Predicting...')
    preds = []
    allpreds = []
    for batch in testA_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            logits = outputs[1]
            logits = torch.softmax(logits, dim=1).cpu().detach().numpy()
            preds = np.argmax(logits, axis=1).tolist()
            allpreds += preds
    print('Done.')

    submission = test
    submission['label'] = allpreds
    submission.drop('text', axis=1, inplace=True)

    submission.to_csv(config.result + config.student_id + 'submission_{}.csv'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                      index=False)

if __name__ == '__main__':
    conf = Config()
    device = torch.device(conf.device)
    model = Bert_Pytorch().to(device)

    predict(model, conf, device)