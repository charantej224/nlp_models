from dataset.data_handler import load_datasets, load_test_datasets
from models.bert.model_train_test import start_epochs, load_model
import os
import pandas as pd
import numpy as np

from utils.file_utils import write_json_dict, read_json

root_dir = "/home/charan/DATA/311_Data/Department/"
final_data = os.path.join(root_dir, "311_VIZ_DESCRIPTION_PARENT.csv")
write_json = os.path.join(root_dir, "class.json")
load_model_path = ""
label_dict = read_json(write_json)


def setup_data(input_data):
    input_data['label'] = input_data.PARENT_DEPT.apply(lambda x: label_dict[x])
    input_data.rename(columns={"CASE ID": "u_id", "Description": "desc"}, inplace=True)
    return input_data


def train_classification():
    classification_df = pd.read_csv(final_data)
    classification_df = setup_data(classification_df)
    number_of_classes = max(list(classification_df['label'].unique())) + 1
    model_directory = os.path.join(root_dir, "classify_dict")
    metrics_json = os.path.join(root_dir, "accuracy_metrics.json")
    training_loader, testing_loader = load_datasets(classification_df, train_size=0.8,
                                                    number_of_classes=number_of_classes)
    unique_ids, val_targets, val_outputs = start_epochs(training_loader, testing_loader, metrics_json, model_directory,
                                                        epochs=20, number_of_classes=number_of_classes)
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


def stat_data():
    viz_data = pd.read_csv(final_data)
    print(viz_data.shape)
    counter = 0
    dict_vis = {}
    for each in list(viz_data.VIS_STREET.unique()):
        dict_vis[str(counter)] = each
        dict_vis[each] = counter
        counter += 1
    write_json_dict(dict_vis, write_json)


if __name__ == '__main__':
    train_classification()

# load_model_path = os.path.join(root_dir, 'prob_state_dict2.pt')
# load_model(load_model_path, training_loader, testing_loader)
