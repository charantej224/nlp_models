import json
from sklearn.metrics import precision_recall_fscore_support


def write_json_dict(input_dict, file_name):
    with open(file_name, 'w') as f:
        json.dump(input_dict, f, indent=2)
        f.close()


def read_json(file_path):
    with open(file_path, 'r') as f:
        val_dict = json.load(f)
        f.close()
        return val_dict


def write_file(file_path, input_text):
    with open(file_path, 'w') as f:
        f.write(input_text)


if __name__ == "__main__":
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 2, 2, 1]
    target_names = ['class 0', 'class 1', 'class 2']
    print(precision_recall_fscore_support(y_true, y_pred))
