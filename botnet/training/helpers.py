import pickle
from csv import writer
import numpy as np
import joblib
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, recall_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy


def create_DNN(units, input_dim_param, lr_param):
    network = Sequential()
    network.add(Input(shape=(input_dim_param,)))  # Generalized shape
    print(units)
    network.add(Dense(units=units[0], activation='relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units=units[1], activation='relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units=units[2], activation='relu'))
    network.add(Dense(units=2, activation='softmax'))  # Softmax for 2 classes

    optimizer = Adam(learning_rate=lr_param)
    network.compile(optimizer=optimizer, loss=CategoricalCrossentropy())
    return network


def save_metrics(y_true, y_pred, metrics_path):
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    metrics_obj = {
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fp / (fp + tn),
        'fnr': fn / (fn + tp)
    }
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_obj, f)


def save_adv_candidates(x_test_array, y_test_array, predictions, adv_candidates_path):
    idx = np.argwhere((predictions == y_test_array) & (y_test_array == 1))
    idx = np.squeeze(idx)
    np.save(adv_candidates_path, x_test_array[idx])


def save_train_data(nn, x_test_array, y_test_array, metrics_path, adv_candidates_path):
    probas = nn.predict(x_test_array)
    predictions = np.argmax(probas, axis=1)  #Choose class with highest probability

    save_metrics(np.argmax(y_test_array, axis=1), predictions, metrics_path)
    save_adv_candidates(x_test_array, np.argmax(y_test_array, axis=1), predictions, adv_candidates_path)


def read_metrics(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def add_data_to_df(data, df_path):
    with open(df_path, 'a', newline="") as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(data)


def get_adversarials(adv_candidates_path, model_path, distance, attack, adv_path):
    perturb_samples_pgd, pgd_success_rate = attack(attack, model_path, adv_candidates_path, distance, 100)
    np.save(adv_path, perturb_samples_pgd)


def read_min_max(min_file, max_file):
    with open(min_file, 'r') as f:
        min_features = [float(i) for i in f.read().strip().replace(' ', '').split(',')]
    with open(max_file, 'r') as f:
        max_features = [float(i) for i in f.read().strip().replace(' ', '').split(',')]
    return min_features, max_features


def get_train_data(config):
    x_train = np.load(config["x_train"])
    y_train = np.load(config["y_train"])
    x_test = np.load(config["x_test"])
    y_test = np.load(config["y_test"])

    #  Convert binary labels to one-hot for softmax compatibility
    if y_train.ndim == 1 or y_train.shape[1] == 1:
        y_train = to_categorical(np.squeeze(y_train), num_classes=2)

    if y_test.ndim == 1 or y_test.shape[1] == 1:
        y_test = to_categorical(np.squeeze(y_test), num_classes=2)

    return x_train, y_train, x_test, y_test


def get_model_data(config):
    return config["LAYERS"], config["INPUT_DIM"], config["LR"]


def get_processing_data(config):
    scaler = joblib.load(config["scaler_path"])
    min_features, max_features = read_min_max(config["min_features"], config["max_features"])
    mask_idx = np.load(config["mask_idx"])
    eq_min_max = np.load(config["eq_min_max_idx"])
    return scaler, min_features, max_features, mask_idx, eq_min_max
