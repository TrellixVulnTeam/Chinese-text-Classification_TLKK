import torch
from utils import BertTokenizer

class Config():
    def __init__(self):

        self.train_path = '/home/chinese_text_cls/data/train.txt' # 训练集
        self.dev_path = '/home/chinese_text_cls/data/dev.txt'     # 验证集
        self.test_path = '/home/chinese_text_cls/data/test.txt'   # 测试集
        self.class_list = [x.strip() for x in
                           open('/home/chinese_text_cls/data/class.txt').readlines()] # 类别名单

        self.model_name = 'bert'
        self.num_classes = len(self.class_list)  # 类别数
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.bert_path = '/home/chinese_text_cls/pretrain'
        self.hidden_size = 768
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')  # 设备
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 30  # epoch数
        self.batch_size = 512  # mini-batch大小
        self.learning_rate = 5e-5  # 学习率

        self.save_path = '/home/chinese_text_cls/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.test_csv = '/home/chinese_text_cls/data/testA.csv'
        self.result = '/home/chinese_text_cls'
        self.student_id = '123'

