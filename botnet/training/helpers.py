import os
import pickle
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

def create_DNN(units, input_dim_param, lr_param):
   
    network = Sequential()
    network.add(Input(shape=(input_dim_param,)))
    network.add(Dense(units=units[0], activation='relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units=units[1], activation='relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units=units[2], activation='relu'))
    network.add(Dense(units=2))  # Output layer for multi-class classification
    network.add(Activation('softmax'))  # Softmax for multi-class classification

    optimizer = Adam(learning_rate=lr_param)
    network.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

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
   
    try:
        with open(min_file, "r") as f:
            mins = list(map(float, f.read().strip().split(",")))
        with open(max_file, "r") as f:
            maxs = list(map(float, f.read().strip().split(",")))
        return mins, maxs
    except Exception as e:
        print(f"[ERROR] Failed to read min/max files: {e}")
        raise

def get_train_data(config):
    
    try:
        x_train = np.load(config["x_train"])
        y_train = np.load(config["y_train"])
        x_test = np.load(config["x_test"])
        y_test = np.load(config["y_test"])
        print(f"[DEBUG] x_train shape: {x_train.shape}")
        print(f"[DEBUG] x_test shape: {x_test.shape}")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        print(f"[ERROR] Failed to load training/testing data: {e}")
        raise

def get_model_data(config):
    return config["LAYERS"], config["INPUT_DIM"], config["LR"]

def get_processing_data(config):
    
    try:
        scaler = joblib.load(config["scaler_path"])
        min_features, max_features = read_min_max(config["min_features"], config["max_features"])
        return scaler, min_features, max_features
    except Exception as e:
        print(f"[ERROR] Failed to load processing data: {e}")
        raise

def save_model_and_history(config, model, history_obj, method):
   
    path_to_save = config["path_to_save"]

    # Save the model in .keras format
    keras_model_path = f"{path_to_save}/{method}_model.keras"
    print(f"[DEBUG] Saving model in .keras format to {keras_model_path}...")
    model.save(keras_model_path)

    # Save the model in .h5 format
    h5_model_path = f"{path_to_save}/{method}_model.h5"
    print(f"[DEBUG] Saving model in .h5 format to {h5_model_path}...")
    model.save(h5_model_path)

    # Save the model in SavedModel format
    saved_model_dir = f"{path_to_save}/{method}_model_saved"
    print(f"[DEBUG] Exporting model in TensorFlow SavedModel format to {saved_model_dir}...")
    tf.saved_model.save(model, saved_model_dir)

    # Save the training history
    history_path = f"{path_to_save}/{method}_model_history.npy"
    print(f"[DEBUG] Saving training history to {history_path}...")
    with open(history_path, "wb") as f:
        np.save(f, dict(history_obj.history))

    print(f"[INFO] Model and history for {method} saved successfully at {path_to_save}")
