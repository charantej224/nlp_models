from torch import cuda
from models.bert.bert_classification import BERTClass
from utils.app_logger import AppLogger
import numpy as np
from sklearn.metrics import accuracy_score
import json, time, os, torch

device = 'cuda' if cuda.is_available() else 'cpu'
logger = AppLogger.getInstance()


def setup_model(no_class_1, no_class_2):
    bert_model = BERTClass(no_class_1, no_class_2)
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
    train_targets = np.array([])
    train_outputs = np.array([])
    counter = 0
    total = len(training_loader)
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        train_targets = np.append(train_targets, np.argmax(targets.cpu().detach().numpy(), axis=1))
        train_outputs = np.append(train_outputs, np.argmax(outputs.cpu().detach().numpy(), axis=1))
        unique_ids = np.append(unique_ids, data['u_id'])
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter = counter + len(data)
        if counter % 100 == 0:
            print(f" Epoch - {epoch} - current training {counter / 8} / {total}")

    torch.save(model.state_dict(), model_directory + '_' + str(epoch) + ".pt")
    done = time.time()
    elapsed = (done - start) / 60
    return unique_ids, train_targets, train_outputs, elapsed


def validation(epoch, testing_loader, model):
    start = time.time()
    model.eval()
    validation_targets = np.array([])
    validation_outputs = np.array([])
    unique_ids = np.array([])
    print(f'Epoch - Inference : {epoch}')
    counter = 0
    total = len(testing_loader)
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            validation_targets = np.append(validation_targets, np.argmax(targets.cpu().numpy(), axis=1))
            validation_outputs = np.append(validation_outputs, np.argmax(outputs.cpu().numpy(), axis=1))
            unique_ids = np.append(unique_ids, data['u_id'])
            counter = counter + len(data)
            if counter % 100 == 0:
                print(f" Epoch - {epoch} - current Inference {counter / 4} / {total}")
    done = time.time()
    elapsed = (done - start) / 60
    return unique_ids, validation_targets, validation_outputs, elapsed


def remove_model_paths(best_epoch, model_path, total_epochs):
    if best_epoch is None:
        return
    best_model_path = model_path + '_' + str(best_epoch) + ".pt"
    for epoch in range(total_epochs):
        current_path = model_path + '_' + str(epoch) + ".pt"
        if os.path.exists(current_path) and current_path != best_model_path:
            os.remove(current_path)


def start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs, no_class_1, no_class_2):
    model = setup_model(no_class_1, no_class_2)
    optimizer = get_optimizer(model)
    accuracy_map = {}
    best_val_accuracy = 0

    for epoch in range(epochs):
        unique_ids, train_targets, train_outputs, train_time = train(epoch, training_loader, model, optimizer,
                                                                     model_directory)
        train_accuracy = accuracy_score(train_targets, train_outputs) * 100
        print('Epoch {} - accuracy {}'.format(epoch, train_accuracy))
        unique_ids, val_targets, val_outputs, inference_time = validation(epoch, testing_loader, model)
        validation_accuracy = accuracy_score(val_targets, val_outputs) * 100
        if validation_accuracy > best_val_accuracy:
            best_unique_ids, best_val_targets, best_val_outputs, best_epoch = unique_ids, val_targets, val_outputs, epoch
            best_val_accuracy = validation_accuracy
        print('Epoch {} - accuracy {}'.format(epoch, validation_accuracy))
        accuracy_map["train_accuracy_" + str(epoch)] = train_accuracy
        accuracy_map["train_time_" + str(epoch)] = train_time
        accuracy_map["val_accuracy_" + str(epoch)] = validation_accuracy
        accuracy_map["val_time_" + str(epoch)] = inference_time

    with open(metrics_json, 'w') as f:
        json.dump(accuracy_map, f, indent=2)
        f.close()
    remove_model_paths(best_epoch, model_directory, epochs)
    return best_unique_ids, best_val_targets, best_val_outputs


def load_model(model_file, testing_loader, number_of_classes):
    model = setup_model(number_of_classes)
    model.load_state_dict(torch.load(model_file))
    unique_ids, val_targets, val_outputs, inference_time = validation(1, testing_loader, model)
    return unique_ids, val_outputs
