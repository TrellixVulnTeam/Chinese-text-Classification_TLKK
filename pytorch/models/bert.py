import torch.nn as nn
from transformers import BertForSequenceClassification

class Bert_Pytorch(nn.Module):
    def __init__(self):
        super(Bert_Pytorch, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=17)

    def forward(self, input_ids, attention_mask, labels):

        return self.bert(input_ids, attention_mask=attention_mask, labels=labels)
