import os
import json
import numpy as np
import tensorflow as tf
from helpers import create_DNN, get_train_data, get_model_data, get_processing_data, save_model_and_history
from datagen import generate_adversarial_batch_pgd, generate_adversarial_batch_fence

def load_mutable_idx(config):
    mutable_idx = config.get("mutable_idx", None)
    if not mutable_idx:
        raise ValueError("The 'mutable_idx' is not specified in the configuration file.")

    if not os.path.exists(mutable_idx):
        raise FileNotFoundError(f"The file 'mutable_idx.npy' does not exist at {mutable_idx}. Please check the path.")

    print(f"[INFO] Loading mutable_idx from: {mutable_idx}")
    mutable_idx = np.load(mutable_idx)
    print("[INFO] Successfully loaded mutable_idx.")
    return mutable_idx

def load_eq_min_max_idx():
    eq_min_max_idx = "../data/neris/eq_min_max_idx.npy"
    if not os.path.exists(eq_min_max_idx):
        raise FileNotFoundError(f"The file 'eq_min_max_idx.npy' does not exist at {eq_min_max_idx}. Please check the path.")

    print(f"[INFO] Loading eq_min_max_idx from: {eq_min_max_idx}")
    eq_min_max_idx = np.load(eq_min_max_idx)
    print("[INFO] Successfully loaded eq_min_max_idx.")
    return eq_min_max_idx

def train(config, method="clean", distance=12, save_data=True):
    # Load data
    x_train, y_train, x_test, y_test = get_train_data(config)
    LAYERS, INPUT_DIM, LR = get_model_data(config)
    scaler, min_features, max_features = get_processing_data(config)

    print(f"[DEBUG] x_train shape: {x_train.shape}")
    print(f"[DEBUG] x_test shape: {x_test.shape}")

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
    epochs = config["epochs"]
    batch_size = 64
    steps_per_epoch = len(x_train) // batch_size

    for lrate in LR:
        model = create_DNN(units=LAYERS, input_dim_param=INPUT_DIM, lr_param=lrate)
        print(f"[INFO] Training model with learning rate: {lrate}...")

        if method == "pgd":
            print("[INFO] Generating PGD adversarial examples...")
            try:
                mutable_idx = load_mutable_idx(config)
                eq_min_max_idx = load_eq_min_max_idx()

                dataGen = generate_adversarial_batch_pgd(
                    model,
                    total=batch_size,
                    samples=x_train,
                    labels=y_train,
                    distance=distance,
                    iterations=config["iterations"],
                    scaler=scaler,
                    mins=min_features,
                    maxs=max_features,
                    mutable_idx=mutable_idx,
                    eq_min_max_idx=eq_min_max_idx
                )
                history_obj = model.fit(dataGen, steps_per_epoch=steps_per_epoch, verbose=1, epochs=epochs)
            except Exception as e:
                print(f"[ERROR] Error during PGD training: {e}")
                continue

        elif method == "fence":
            print("[INFO] Generating FENCE adversarial examples...")
            try:
                dataGen = generate_adversarial_batch_fence(
                    model=model,
                    total=batch_size,
                    samples=x_train,
                    labels=y_train,
                    distance=distance,
                    iterations=config["iterations"],
                    scaler=scaler,
                    mins=min_features,
                    maxs=max_features,
                    model_path=config["intermediate_model_path"],
                )
                history_obj = model.fit(dataGen, steps_per_epoch=steps_per_epoch, verbose=1, epochs=epochs)
            except Exception as e:
                print(f"[ERROR] Error during FENCE training: {e}")
                continue

        if save_data:
            save_model_and_history(config, model, history_obj, method)

if __name__ == "__main__":
    config_file = "../training/config/neris.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    attack_methods = config["attack_methods"]
    for method in attack_methods:
        print(f"[INFO] Training with method: {method}...")
        distances = config["distances"]
        for distance in distances:
            print(f"[INFO] Training with distance: {distance}...")
            try:
                train(config, method=method, distance=int(distance), save_data=True)
            except Exception as e:
                print(f"[ERROR] Error during training with method {method} and distance {distance}: {e}")
