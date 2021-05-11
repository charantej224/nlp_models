from dataset.custom_data import CustomDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def load_datasets(classification_dataframe, train_size, no_class_1, no_class_2):
    train_dataset = classification_dataframe.sample(frac=train_size, random_state=200)
    test_dataset = classification_dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(classification_dataframe.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    for each in list(train_dataset.label2.unique()):
        each_df = train_dataset[train_dataset.label2 == each]
        print("Train for class {} Dataset: {}".format(each, each_df.shape))
    for each in list(test_dataset.label2.unique()):
        each_df = test_dataset[test_dataset.label2 == each]
        print("Test for class {} Dataset: {}".format(each, each_df.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, no_class_1, no_class_2)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN, no_class_1, no_class_2)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader


def load_test_datasets(test_df, number_of_classes=16):
    test_df = test_df.reset_index(drop=True)
    print("Test Dataset: {}".format(test_df.shape))
    testing_set = CustomDataset(test_df, tokenizer, MAX_LEN, number_of_classes, inference=True)

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    testing_loader = DataLoader(testing_set, **test_params)
    return testing_loader
