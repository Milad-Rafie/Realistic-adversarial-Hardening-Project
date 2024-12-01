import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.path.append('..')
import joblib
from datetime import datetime
import numpy as np
import tensorflow as tf
from fence.neris_attack_tf2 import Neris_attack
from pgd.pgd_attack_art import PgdRandomRestart
from training.helpers import read_min_max
from tensorflow.random import set_seed
from tensorflow.keras.losses import BinaryCrossentropy
import json

# Helper to load mutable indices
def load_mutable_idx(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file '{path}' does not exist. Please check the path.")
    print(f"[INFO] Loading mutable_idx from: {path}")
    return np.load(path)

# Helper to evaluate success rate
def evaluate_success_rate(model, perturbSamples, y_test):
    probas = np.squeeze(model.predict(perturbSamples))
    predictions = np.argmax(probas, axis=1)
    adv_success = predictions != np.argmax(y_test, axis=1)  # Success if prediction changes
    success_rate = np.mean(adv_success) * 100
    return success_rate

# Main attack function
def attack(config, method="clean", distance=12):
    print("[INFO] Loading test samples and labels...")
    samples = np.load(config["x_test"])
    labels = np.load(config["y_test"])
    labels = np.squeeze(labels)
    print(f"[DEBUG] Samples shape: {samples.shape}, Labels shape: {labels.shape}")

    if config.get("only_botnet", True):
        idx = np.where(labels == 1)[0]
        samples = samples[idx]
        labels = labels[idx]
        print(f"[DEBUG] Filtered botnet samples shape: {samples.shape}, Labels shape: {labels.shape}")

    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)
    print(f"[DEBUG] One-hot encoded labels shape: {labels_one_hot.shape}")

    print(f"[INFO] Loading model from {config['intermediate_model_path']}...")
    try:
        model = tf.keras.models.load_model(config["intermediate_model_path"])
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None, None

    perturbSamples = None
    success_rate = 0

    if method == "clean":
        print(f"[INFO] Running clean evaluation...")
        perturbSamples = samples
        success_rate = evaluate_success_rate(model, perturbSamples, labels_one_hot)

    elif method == "pgd":
        print(f"[INFO] Generating PGD adversarial examples...")
        mutable_idx = load_mutable_idx(config["mutable_idx"])
        eq_min_max_idx = load_mutable_idx(config["eq_min_max_idx"])
        scaler = joblib.load(config["scaler_path"])
        min_features, max_features = read_min_max(config["min_features"], config["max_features"])
        attack_generator = PgdRandomRestart(
            model=model,
            eps=distance,
            alpha=1,
            num_iter=config["iterations"],
            restarts=5,
            scaler=scaler,
            mins=min_features,
            maxs=max_features,
            mutable_idx=mutable_idx,
            eq_min_max_idx=eq_min_max_idx
        )
        try:
            perturbSamples = attack_generator.run_attack(samples, labels_one_hot)
            success_rate = evaluate_success_rate(model, perturbSamples, labels_one_hot)
        except Exception as e:
            print(f"[ERROR] Error during PGD attack: {e}")

    elif method == "fence":
        print(f"[INFO] Generating FENCE adversarial examples...")
        scaler = joblib.load(config["scaler_path"])
        min_features, max_features = read_min_max(config["min_features"], config["max_features"])
        attack_generator = Neris_attack(
            model_path=config["intermediate_model_path"],
            iterations=config["iterations"],
            distance=distance,
            scaler=scaler,
            mins=min_features,
            maxs=max_features
        )
        perturbSamples = []
        try:
            for i in range(samples.shape[0]):
                if i % 100 == 0:
                    print(f"[INFO] Processing sample {i}/{samples.shape[0]}...")
                sample = np.expand_dims(samples[i], axis=0)
                label = labels[i]
                adversary = attack_generator.run_attack(sample, label)
                perturbSamples.append(adversary)
            perturbSamples = np.squeeze(np.array(perturbSamples))
            success_rate = evaluate_success_rate(model, perturbSamples, labels_one_hot)
        except Exception as e:
            print(f"[ERROR] Error during FENCE attack: {e}")
    else:
        print(f"[ERROR] Unsupported attack method: {method}")

    print(f"[INFO] {method.upper()} attack success rate: {success_rate:.2f}%")
    return perturbSamples, success_rate

if __name__ == "__main__":
    config_file = "../config/neris.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    # Set random seeds for reproducibility
    np.random.seed(500)
    set_seed(500)

    start_time = datetime.now()
    attack_methods = config["attack_methods"]
    distances = [int(d) for d in config["distances"]]

    for method in attack_methods:
        for distance in distances:
            print(f"\n[INFO] Running {method.upper()} attack with distance {distance}...")
            try:
                perturbed_samples, success_rate = attack(config, method=method, distance=distance)
                if perturbed_samples is not None:
                    print(f"[RESULT] Success rate for {method.upper()} with distance {distance}: {success_rate:.2f}%")
            except Exception as e:
                print(f"[ERROR] Error during {method.upper()} attack with distance {distance}: {e}")

    end_time = datetime.now()
    print(f"Total Duration: {end_time - start_time}")
