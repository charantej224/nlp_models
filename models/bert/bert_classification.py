import torch
import transformers
from transformers import BertTokenizer
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


class BERTClass(torch.nn.Module):
    def __init__(self, no_class_1, no_class_2, label_cat):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        super(BERTClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.classifier1 = BERTClass.get_classifier(no_class_1)
        self.classifier2 = BERTClass.get_classifier(no_class_2)
        self.label_cat = label_cat

    def forward(self, ids, mask, token_type_ids, desc):
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_csl1 = self.classifier1(output_1)
        level_ids, level_mask, level_token = self.get_inputs(output_csl1, desc)
        _, output_2 = self.bert(level_ids, attention_mask=level_mask, token_type_ids=level_token)
        output_csl2 = self.classifier2(output_2)
        return output_csl1, output_csl2

    @staticmethod
    def get_classifier(no_of_classes):
        return torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, 512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, no_of_classes),
            torch.nn.Softmax()
        )

    def get_inputs(self, output_args, list_desc):
        vals = torch.argmax(output_args, dim=1)
        counter = vals.shape[0]
        list_ids, list_mask, list_token = [], [], []
        for each in range(counter):
            category = self.label_cat[str(vals[each].item())]
            ids, mask, token = self.get_ids(category + " " + list_desc[each])
            ten_ids, ten_mask, ten_token = list_ids.append(ids), list_mask.append(mask), list_token.append(token)
        ten_ids, ten_mask, ten_token = torch.stack(list_ids), torch.stack(list_mask), torch.stack(list_token)
        ten_ids, ten_mask, ten_token = ten_ids.to(device, dtype=torch.long), ten_mask.to(device,
                                                                                         dtype=torch.long), ten_token.to(device, dtype=torch.long)
        return ten_ids, ten_mask, ten_token

    def get_ids(self, desc_text):
        inputs = self.tokenizer.encode_plus(
            desc_text,
            None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids,
                                                                                                       dtype=torch.long)
