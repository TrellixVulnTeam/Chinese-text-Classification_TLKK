import torch
from torch.utils.data import DataLoader
from config import Config
from datasets.utils import split_train_val, text_encoding, NewsDataset
from models.bert import Bert_Pytorch
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from torch.utils.tensorboard import SummaryWriter
import os
import tempfile
from transformers import AdamW
import math
import torch.optim.lr_scheduler as lr_scheduler
from tools.train_eval_utils import train_one_epoch, evaluate

# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()

    # 初始化各进程环境
    init_distributed_mode(args=conf)

    rank = conf.rank
    device = torch.device(conf.device)
    batch_size = conf.batch_size
    conf.lr *= conf.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
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

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    # dataloader
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_batch_sampler,
                              pin_memory=True,
                              num_workers=nw)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            pin_memory=True,
                            num_workers=nw)

    # model
    model = Bert_Pytorch().to(device)
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[conf.gpu])

    # optimizer
    optimizer = AdamW(model.parameters(), conf.lr)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / conf.epochs)) / 2) * (1 - conf.lrf) + conf.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train
    for epoch in range(conf.epochs):
        train_sampler.set_epoch(epoch)

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

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.module.state_dict(), conf.save_dict.format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()