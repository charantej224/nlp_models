from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, no_classes1, no_classes2, inference=False):
        self.tokenizer = tokenizer
        self.features = dataframe.desc
        self.inference = inference
        if not self.inference:
            self.label1 = dataframe.label1
            self.label2 = dataframe.label2
        self.unique_ids = dataframe.u_id
        self.max_len = max_len
        self.no_classes1 = no_classes1
        self.no_classes2 = no_classes2

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        desc_text = str(self.features[index])
        desc_text = " ".join(desc_text.split())

        inputs = self.tokenizer.encode_plus(
            desc_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        target1 = torch.zeros(self.no_classes1)
        target2 = torch.zeros(self.no_classes2)
        if not self.inference:
            target1[self.label1[index]] = 1
            target2[self.label2[index]] = 1

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target1': target1,
            'target2': target2,
            'u_id': self.unique_ids[index],
            'text': desc_text
        }
