import torch
import transformers


class BERTClass(torch.nn.Module):
    def __init__(self, no_class_1, no_class_2):
        super(BERTClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.classifier1 = self.get_classifier(no_class_1)
        self.classifier2 = self.get_classifier(no_class_2)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.classifier1(output_1)
        return output

    def get_classifier(self, no_of_classes):
        classifer_layer = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, 512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, no_of_classes),
            torch.nn.Softmax()
        )
        return classifer_layer
