from torch import cuda
from models.bert.bert_classification import BERTClass
from utils.app_logger import AppLogger
import numpy as np
from sklearn.metrics import accuracy_score
import json, time, os, torch
import pandas as pd

device = 'cuda' if cuda.is_available() else 'cpu'
logger = AppLogger.getInstance()
root_dir = "/home/charan/DATA/311_Data/multi-level-classification"


def setup_model(no_class_1, no_class_2, label_cat, label_type):
    bert_model = BERTClass(no_class_1, no_class_2, label_cat)
    bert_model.to(device)
    return bert_model


def get_optimizer(bert_model):
    learning_rate = 1e-05
    adam_optimizer = torch.optim.Adam(params=bert_model.parameters(), lr=learning_rate)
    return adam_optimizer


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(epoch, training_loader, model, optimizer, model_directory):
    logger.debug("training started")
    start = time.time()
    model.train()
    unique_ids = np.array([])
    train_targets_1 = np.array([])
    train_outputs_1 = np.array([])
    train_targets_2 = np.array([])
    train_outputs_2 = np.array([])
    counter = 0
    total = len(training_loader)
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        target1 = data['target1'].to(device, dtype=torch.float)
        target2 = data['target2'].to(device, dtype=torch.float)
        desc = data['text']
        output1, output2 = model(ids, mask, token_type_ids, desc)
        optimizer.zero_grad()
        loss1 = loss_fn(output1, target1)
        loss2 = loss_fn(output2, target2)
        loss = loss1 + loss2
        train_targets_1 = np.append(train_targets_1, np.argmax(target1.cpu().detach().numpy(), axis=1))
        train_outputs_1 = np.append(train_outputs_1, np.argmax(output1.cpu().detach().numpy(), axis=1))
        train_targets_2 = np.append(train_targets_2, np.argmax(target2.cpu().detach().numpy(), axis=1))
        train_outputs_2 = np.append(train_outputs_2, np.argmax(output2.cpu().detach().numpy(), axis=1))
        unique_ids = np.append(unique_ids, data['u_id'])
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter = counter + len(data)
        if counter % 100 == 0:
            print(f" Epoch - {epoch} - current training {counter / 4} / {total}")

    torch.save(model.state_dict(), model_directory + '_' + str(epoch) + ".pt")
    done = time.time()
    elapsed = (done - start) / 60
    return unique_ids, train_targets_1, train_outputs_1, train_targets_2, train_outputs_2, elapsed


def validation(epoch, testing_loader, model):
    start = time.time()
    model.eval()
    val_targets_1 = np.array([])
    val_outputs_1 = np.array([])
    val_targets_2 = np.array([])
    val_outputs_2 = np.array([])
    unique_ids = np.array([])
    print(f'Epoch - Inference : {epoch}')
    counter = 0
    total = len(testing_loader)
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            target1 = data['target1'].to(device, dtype=torch.float)
            target2 = data['target2'].to(device, dtype=torch.float)
            desc = data['text']
            output1, output2 = model(ids, mask, token_type_ids, desc)
            val_targets_1 = np.append(val_targets_1, np.argmax(target1.cpu().numpy(), axis=1))
            val_outputs_1 = np.append(val_outputs_1, np.argmax(output1.cpu().numpy(), axis=1))
            val_targets_2 = np.append(val_targets_2, np.argmax(target2.cpu().numpy(), axis=1))
            val_outputs_2 = np.append(val_outputs_2, np.argmax(output2.cpu().numpy(), axis=1))
            unique_ids = np.append(unique_ids, data['u_id'])
            counter = counter + len(data)
            if counter % 100 == 0:
                print(f" Epoch - {epoch} - current Inference {counter / 4} / {total}")
    done = time.time()
    elapsed = (done - start) / 60
    return unique_ids, val_targets_1, val_outputs_1, val_targets_2, val_outputs_2, elapsed


def remove_model_paths(best_epoch, model_path, total_epochs):
    if best_epoch is None:
        return
    best_model_path = model_path + '_' + str(best_epoch) + ".pt"
    for epoch in range(total_epochs):
        current_path = model_path + '_' + str(epoch) + ".pt"
        if os.path.exists(current_path) and current_path != best_model_path:
            os.remove(current_path)


def start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs, no_class_1, no_class_2,
                 label_cat, label_type):
    model = setup_model(no_class_1, no_class_2, label_cat, label_type)
    optimizer = get_optimizer(model)
    accuracy_map = {}
    for epoch in range(epochs):
        predicted = os.path.join(root_dir, f"predicted_{epoch}.csv")
        unique_ids, targets_1, outputs_1, targets_2, outputs_2, elapsed = train(epoch, training_loader, model,
                                                                                optimizer,model_directory)
        cls1_train_accuracy = accuracy_score(targets_1, outputs_1) * 100
        cls2_train_accuracy = accuracy_score(targets_2, outputs_2) * 100
        print(f'Epoch {epoch} - classifier1 {cls1_train_accuracy} % -classifier2 {cls2_train_accuracy} %')
        val_unique_ids, val_target1, val_outputs1, val_target2, val_outputs2, val_elapsed = validation(epoch,
                                                                                                       testing_loader,
                                                                                                       model)
        cls1_validation_accuracy = accuracy_score(val_target1, val_outputs1) * 100
        cls2_validation_accuracy = accuracy_score(val_target2, val_outputs2) * 100
        print(f'Epoch {epoch} - val classifier1 {cls1_validation_accuracy} val classifier2 {cls2_validation_accuracy}')
        accuracy_map[f"train_cls1_{str(epoch)}"] = cls1_train_accuracy
        accuracy_map[f"train_cls2_{str(epoch)}"] = cls2_train_accuracy
        accuracy_map[f"val_cls1_{str(epoch)}"] = cls1_validation_accuracy
        accuracy_map[f"val_cls2_{str(epoch)}"] = cls2_validation_accuracy
        accuracy_map[f"train_time_{str(epoch)}"] = elapsed
        accuracy_map[f"val_time_{str(epoch)}"] = val_elapsed
        val_unique_ids, val_target1, val_outputs1, val_target2, val_outputs2 = val_unique_ids.reshape(-1, 1), val_target1.reshape(-1, 1), val_outputs1.reshape(-1, 1), val_target2.reshape(-1, 1), val_outputs2.reshape(-1, 1)

        out_numpy = np.concatenate((val_unique_ids, val_target1, val_outputs1, val_target2, val_outputs2), axis=1)
        predicted_df = pd.DataFrame(out_numpy, columns=['id', 'original_cls1', 'predicted_cls1', 'original_cls2',
                                                        'predicted_cls2'])
        predicted_df.to_csv(predicted, index=False)

    with open(metrics_json, 'w') as f:
        json.dump(accuracy_map, f, indent=2)
        f.close()


def load_model(model_file, testing_loader, number_of_classes):
    model = setup_model(number_of_classes)
    model.load_state_dict(torch.load(model_file))
    unique_ids, val_targets, val_outputs, inference_time = validation(1, testing_loader, model)
    return unique_ids, val_outputs
