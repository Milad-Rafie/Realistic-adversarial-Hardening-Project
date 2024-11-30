import os
import pickle
from csv import writer
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
)


def create_DNN(units, input_dim_param, lr_param):
    network = Sequential()
    network.add(Input(shape=(input_dim_param,)))
    network.add(Dense(units=units[0], activation='relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units=units[1], activation='relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units=units[2], activation='relu'))
    network.add(Dense(units=2))  # Changed to 2 units for multi-class output
    network.add(Activation('softmax'))  # Use softmax for multi-class classification

    sgd = Adam(learning_rate=lr_param)
    network.compile(optimizer=sgd, loss=CategoricalCrossentropy(), metrics=['accuracy'])

    return network


def save_metrics(y_test_array, predictions, metrics_path):
    f1 = f1_score(np.argmax(y_test_array, axis=1), predictions)
    roc_auc = roc_auc_score(np.argmax(y_test_array, axis=1), predictions)
    tn, fp, fn, tp = confusion_matrix(np.argmax(y_test_array, axis=1), predictions).ravel()
    metrics_obj = {
        "f1": f1,
        "roc_auc": roc_auc,
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
    }
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics_obj, f)


def save_adv_candidates(x_test_array, y_test_array, predictions, adv_candidates_path):
    idx = np.argwhere((predictions != np.argmax(y_test_array, axis=1)) & (np.argmax(y_test_array, axis=1) == 1))
    idx = np.squeeze(idx)
    np.save(adv_candidates_path, x_test_array[idx])


def read_min_max(min_file, max_file):
    with open(min_file, "r") as f:
        mins = list(map(float, f.read().strip().split(",")))

    with open(max_file, "r") as f:
        maxs = list(map(float, f.read().strip().split(",")))

    return mins, maxs


def get_train_data(config):
    x_train = np.load(config["x_train"])
    y_train = np.load(config["y_train"])
    x_test = np.load(config["x_test"])
    y_test = np.load(config["y_test"])
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    return x_train, y_train, x_test, y_test


def get_model_data(config):
    return config["LAYERS"], config["INPUT_DIM"], config["LR"]


def get_processing_data(config):
    scaler = joblib.load(config["scaler_path"])
    min_features, max_features = read_min_max(config["min_features"], config["max_features"])
    return scaler, min_features, max_features