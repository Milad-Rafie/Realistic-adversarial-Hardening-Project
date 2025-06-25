import sys
import os
import random as rn
import json
import numpy as np
import tensorflow as tf
import pickle
from datagen import generate_adversarial_batch_pgd
from helpers import create_DNN, save_metrics, save_adv_candidates, get_train_data, get_model_data, get_processing_data
from attack.pgd.pgd_attack_art import PgdRandomRestart
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


def evaluate_model(model, x_test, y_test_onehot, attack_generator=None):
    # Convert one-hot back to integer labels
    y_true = np.argmax(y_test_onehot, axis=1)
    # Clean evaluation
    proba = model.predict(x_test)
    y_pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, proba[:, 1])
    except Exception:
        auc = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("\n=== Clean Test Performance ===")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc if auc is not None else 'n/a'}")
    print(f"FPR: {fp/(fp+tn):.4f}, FNR: {fn/(fn+tp):.4f}")

    # Adversarial evaluation
    if attack_generator is not None:
        # Prepare integer labels for attack
        y_int = y_true.reshape(-1, 1)
        adv_x = attack_generator.run_attack(x_test, y_int)
        proba_adv = model.predict(adv_x)
        y_adv_pred = np.argmax(proba_adv, axis=1)
        acc_adv = accuracy_score(y_true, y_adv_pred)
        f1_adv = f1_score(y_true, y_adv_pred)
        try:
            auc_adv = roc_auc_score(y_true, proba_adv[:, 1])
        except Exception:
            auc_adv = None
        tn, fp, fn, tp = confusion_matrix(y_true, y_adv_pred).ravel()
        print("\n=== Adversarial Test Performance ===")
        print(f"Accuracy: {acc_adv:.4f}, F1: {f1_adv:.4f}, ROC-AUC: {auc_adv if auc_adv is not None else 'n/a'}")
        print(f"FPR: {fp/(fp+tn):.4f}, FNR: {fn/(fn+tp):.4f}")
    else:
        print("No adversarial evaluation performed.")
    print("\n")


def train(config, method='clean', callback=None, distance=12, save_data=True):
    x_train, y_train, x_test, y_test = get_train_data(config)
    LAYERS, INPUT_DIM, LR = get_model_data(config)
    scaler, min_features, max_features, mask_idx, eq_min_max = get_processing_data(config)
    iterations = config["iterations"]
    epochs = config["epochs"]

    for lrate in LR:
        nn = create_DNN(units=LAYERS, input_dim_param=INPUT_DIM, lr_param=lrate)

        if method == "clean":
            history_obj = nn.fit(x_train, y_train, verbose=1, epochs=epochs, batch_size=64, shuffle=True)
        elif method == "pgd":
            dataGen = generate_adversarial_batch_pgd(
                nn, 64, x_train, y_train,
                distance, iterations,
                scaler, min_features, max_features,
                mask_idx, eq_min_max
            )
            history_obj = nn.fit(
                dataGen,
                steps_per_epoch=(len(x_train) // 2) // 64,
                verbose=1,
                epochs=epochs,
                callbacks=callback
            )
        else:
            # fence or other methods
            history_obj = None

        if save_data and history_obj is not None:
            # Save model
            model_path = config["path_to_save"] + f"/{method}_model.h5"
            nn.save(model_path)

            # Save history
            history_path = config["path_to_save"] + f"/{method}_history.npy"
            with open(history_path, 'wb') as f:
                pickle.dump(history_obj.history, f)

            # Evaluate
            # Instantiate attack generator for adversarial evaluation
            attack_gen = PgdRandomRestart(
                model=nn,
                eps=distance,
                alpha=1,
                num_iter=iterations,
                restarts=5,
                scaler=scaler,
                mins=min_features,
                maxs=max_features,
                mutable_idx=mask_idx,
                eq_min_max_idx=eq_min_max
            )
            evaluate_model(nn, x_test, y_test, attack_generator=attack_gen)


def train_save_epochs(config_file, attack):
    with open(config_file) as f:
        config = json.load(f)
    distances = config["distances"]
    for dis in distances:
        cp_path = config["path_to_save"] + f"/distance_{dis}/model-{{epoch:04d}}.h5"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_path,
            save_freq='epoch',
            save_weights_only=False,
            save_best_only=False,
            verbose=1
        )
        train(config, method=attack, callback=cp_callback, distance=int(dis), save_data=True)


if __name__ == "__main__":
    config_file = "config/neris.json"
    attack = "pgd"
    train_save_epochs(config_file, attack)
