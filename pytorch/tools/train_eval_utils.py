import torch
from tools.acc import flat_accuracy
from tqdm import tqdm
import sys
from multi_train_utils.distributed_utils import reduce_value, is_main_process

def train_one_epoch(model, train_loader, optimizer, epoch, device):

    model.train()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        train_loader = tqdm(train_loader, file=sys.stdout)

    for step, batch in enumerate(train_loader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs[0]

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度截断

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            train_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # 参数更新
        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()

@torch.no_grad()
def evaluate(model, val_dataloader, device):

    model.eval()

    # 在进程0中打印验证进度
    if is_main_process():
        val_dataloader = tqdm(val_dataloader, file=sys.stdout)

    total_eval_accuracy = 0
    for step, batch in enumerate(val_dataloader):
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    # total_eval_accuracy = reduce_value(total_eval_accuracy, average=False) # 汇总ACC

    return total_eval_accuracy