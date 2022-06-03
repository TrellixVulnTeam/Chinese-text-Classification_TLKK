# coding: UTF-8
import time
import torch
import numpy as np
from tools.train_eval_utils import train
from tools.utils import get_time_dif
from datasets.build import build_dataset, build_iterator
from config import Config
from models.bert import Bert_FT


if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    cfg = Config()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, _ = build_dataset(cfg)
    train_iter = build_iterator(train_data, cfg)
    dev_iter = build_iterator(dev_data, cfg)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = Bert_FT(cfg).to(cfg.device)
    train(cfg, model, train_iter, dev_iter)