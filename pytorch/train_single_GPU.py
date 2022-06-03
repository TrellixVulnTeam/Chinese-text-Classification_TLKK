import torch
from torch.utils.data import DataLoader
from config import Config
from datasets.utils import split_train_val, text_encoding, NewsDataset
from models.bert import Bert_Pytorch
from torch.utils.tensorboard import SummaryWriter
import os
from transformers import AdamW
import math
import torch.optim.lr_scheduler as lr_scheduler
from tools.train_eval_utils import train_one_epoch, evaluate

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()

    device = torch.device(conf.device)
    rank = conf.rank
    batch_size = conf.batch_size
    checkpoint_path = ""

    # print(conf.__dict__)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # dataset
    x_train, train_label, x_val, val_label, x_testA, _ = split_train_val(conf)
    train_encoding, val_encoding, _ = text_encoding(x_train, x_val, x_testA)
    train_dataset = NewsDataset(train_encoding, train_label)
    val_dataset = NewsDataset(val_encoding, val_label)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    # model
    model = Bert_Pytorch().to(device)

    # optimizer
    optimizer = AdamW(model.parameters(), conf.lr)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / conf.epochs)) / 2) * (1 - conf.lrf) + conf.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train
    for epoch in range(conf.epochs):

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        total_acc = evaluate(model=model,
                             val_dataloader=val_loader,
                             device=device)
        acc = total_acc / len(val_loader)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model, conf.save_dict.format(epoch))