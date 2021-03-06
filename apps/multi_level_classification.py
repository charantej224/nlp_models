from dataset.data_handler import load_datasets, load_test_datasets
from models.bert.model_train_test import start_epochs, load_model
import os
import pandas as pd
import numpy as np

from utils.file_utils import write_json_dict, read_json

# root_dir = "/home/charan/DATA/311_Data/multi-level-classification"
# final_data = os.path.join(root_dir, "balanced_multi-level.csv")
# cat_json = os.path.join(root_dir, "category_class.json")
# type_json = os.path.join(root_dir, "type_class.json")
# load_model_path = ""
# label_cat = read_json(cat_json)
# label_type = read_json(type_json)

root_dir = "/home/charan/DATA/Data/DB_Pedia/archive/multi_level_classification"
# root_dir = "/home/charan/DATA/311_Data/multi-level-feature-extracted"
final_data = os.path.join(root_dir, "DBP_wiki_data_scaled_updated.csv")
final_data_updated = os.path.join(root_dir, "DBP_wiki_data_scaled_updated.csv")
l1_json = os.path.join(root_dir, "l1.json")
l2_json = os.path.join(root_dir, "l2.json")
load_model_path = "/home/charan/DATA/Data/DB_Pedia/archive/multi_level_classification/classify_dict_18.pt"
l1_json = read_json(l1_json)
l2_json = read_json(l2_json)


def get_classes(input_dict):
    counter = 0
    while str(counter) in input_dict:
        counter += 1
    return counter


def setup_data(input_data):
    input_data['label1'] = input_data.PARENT_CATEGORY.apply(lambda x: l1_json[x])
    input_data['label2'] = input_data.TYPE.apply(lambda x: l2_json[x])
    input_data['u_id'] = input_data.index
    input_data.rename(columns={"Description": "desc"}, inplace=True)
    return input_data


def train_classification():
    classification_df = pd.read_csv(final_data)
    classification_df = setup_data(classification_df)
    no_class_1 = get_classes(l1_json)
    no_class_2 = get_classes(l2_json)
    model_directory = os.path.join(root_dir, "classify_dict")
    metrics_json = os.path.join(root_dir, "accuracy_metrics.json")
    training_loader, testing_loader = load_datasets(classification_df, 0.8, no_class_1, no_class_2)
    unique_ids, val_targets, val_outputs = start_epochs(training_loader, testing_loader, metrics_json, model_directory,
                                                        20, no_class_1, no_class_2, l1_json, l2_json)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), val_targets.reshape(-1, 1), val_outputs.reshape(-1, 1)),
                               axis=1)
    predicted_df = pd.DataFrame(out_numpy, columns=['id', 'original', 'predicted'])
    predicted_df.to_csv(os.path.join(root_dir, "predicted.csv"), index=False, header=True)


def inference_classification():
    inference_df = pd.read_csv(final_data_updated)
    no_class_1 = get_classes(l1_json)
    no_class_2 = get_classes(l2_json)
    training_loader, testing_loader = load_datasets(inference_df, 0.8, no_class_1, no_class_2)
    unique_ids, val_targets_1, val_outputs_1, val_targets_2, val_outputs_2 = load_model(load_model_path, testing_loader,
                                                                                        no_class_1, no_class_2, l1_json,
                                                                                        l2_json)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), val_targets_1.reshape(-1, 1), val_outputs_1.reshape(-1, 1),
                                val_targets_2.reshape(-1, 1), val_outputs_2.reshape(-1, 1)), axis=1)
    output_df = pd.DataFrame(out_numpy,
                             columns=['id', 'original_cls1', 'predicted_cls1', 'original_cls2', 'predicted_cls2'])
    output_df.to_csv(os.path.join(root_dir, "predicted.csv"), index=False, header=True)


if __name__ == '__main__':
    inference_classification()
