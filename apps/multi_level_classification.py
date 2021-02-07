from dataset.data_handler import load_datasets, load_test_datasets
from models.bert.model_train_test import start_epochs, load_model
import os
import pandas as pd
import numpy as np

from utils.file_utils import write_json_dict, read_json

root_dir = "/home/charan/DATA/311_Data/multi-level-classification"
final_data = os.path.join(root_dir, "balanced_multi-level.csv")
cat_json = os.path.join(root_dir, "category_class.json")
type_json = os.path.join(root_dir, "type_class.json")
load_model_path = ""
label_cat = read_json(cat_json)
label_type = read_json(type_json)


def get_classes(input_dict):
    counter = 0
    while str(counter) in input_dict:
        counter += 1
    return counter


def setup_data(input_data):
    input_data['label1'] = input_data.PARENT_CATEGORY.apply(lambda x: label_cat[x])
    input_data['label2'] = input_data.TYPE.apply(lambda x: label_type[x])
    input_data['u_id'] = input_data.index
    input_data.rename(columns={"Description": "desc"}, inplace=True)
    return input_data


def train_classification():
    classification_df = pd.read_csv(final_data)
    classification_df = setup_data(classification_df)
    no_class_1 = get_classes(label_cat)
    no_class_2 = get_classes(label_type)
    model_directory = os.path.join(root_dir, "classify_dict")
    metrics_json = os.path.join(root_dir, "accuracy_metrics.json")
    training_loader, testing_loader = load_datasets(classification_df, 0.8, no_class_1, no_class_2)
    unique_ids, val_targets, val_outputs = start_epochs(training_loader, testing_loader, metrics_json, model_directory,
                                                        20, no_class_1, no_class_2, label_cat, label_type)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), val_targets.reshape(-1, 1), val_outputs.reshape(-1, 1)),
                               axis=1)
    predicted_df = pd.DataFrame(out_numpy, columns=['id', 'original', 'predicted'])
    predicted_df.to_csv(os.path.join(root_dir, "predicted.csv"), index=False, header=True)


def inference_classification():
    inference_df = pd.read_csv(final_data)
    test_loader = load_test_datasets(inference_df, 3)
    unique_ids, predictions = load_model(load_model_path, test_loader, 3)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['id', 'classification'])
    dept_df.to_csv(os.path.join(root_dir, "news_data/processed_sentiments.csv"), index=False, header=True)


if __name__ == '__main__':
    train_classification()

# load_model_path = os.path.join(root_dir, 'prob_state_dict2.pt')
# load_model(load_model_path, training_loader, testing_loader)
