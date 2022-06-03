import pandas as pd
import datetime
import torch
import time
import numpy as np
from tools.utils import get_time_dif
from datasets.build import build_dataset, build_iterator
from config import Config
from models.bert import Bert_FT

def predict(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()

    print('Predicting...')
    predict_all = np.array([], dtype=int)
    for text, label in test_iter:
        with torch.no_grad():
            outputs = model(text)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    print('Done.')
    allpreds = list(predict_all)

    test = pd.read_csv(config.test_csv)
    submission = test
    submission['label'] = allpreds
    submission.drop('text', axis=1, inplace=True)
    submission.to_csv(config.result + config.student_id + 'submission_{}.csv'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                      index=False)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':

    cfg = Config()
    _, _, test_data = build_dataset(cfg)
    test_iter = build_iterator(test_data, cfg)

    model = Bert_FT(cfg).to(cfg.device)

    predict(cfg, model, test_iter)
