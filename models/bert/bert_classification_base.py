import torch
from transformers import BertForPreTraining


class BERTClass(torch.nn.Module):
    def __init__(self, number_of_classes=16):
        super(BERTClass, self).__init__()
        # self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        # self.reset_weight(self.bert)
        self.bert = BertForPreTraining.from_pretrained('bert-base-uncased',return_dict=False)
        self.bert.cls.seq_relationship = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, 512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, number_of_classes),
            torch.nn.Softmax()
        )

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # output = self.classifier(output_1)
        return output_1
