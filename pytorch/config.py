import torch

class Config():
    def __init__(self):

        self.train_data = 'data/train.csv'
        self.test_data = 'data/testA.csv'

        self.batch_size = 512
        self.lr = 2e-5
        self.epochs = 10
        self.lrf = 0.1

        self.device = 'cuda'
        self.rank = 0
        self.world_size = 1
        self.gpu = 5
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'

        self.save_dict = "./save_dict/model-{}.pth"
        self.result = './'
        self.student_id = '123'