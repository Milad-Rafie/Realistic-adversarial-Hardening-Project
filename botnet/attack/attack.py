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

# Define a custom loss function to fix reduction='auto'
def custom_loss(*args, **kwargs):
    return BinaryCrossentropy(reduction='sum_over_batch_size')

# Configure seeds for reproducibility
np.random.seed(500)
set_seed(500)
import warnings
warnings.filterwarnings('ignore')

# Main attack function
def attack(method, model_path, samples_path, labels_path, distance, iterations, mask_idx, eq_min_max, only_botnet=True):
    print("Loading samples and labels...")

    # Load samples and labels
    samples = np.load(samples_path)
    labels = np.load(labels_path)
    print(f"[DEBUG] Labels original shape: {labels.shape}")

    # Ensure labels have the correct shape
    labels = np.squeeze(labels)
    print(f"[DEBUG] Labels after squeeze: {labels.shape}")

    # Filter only botnet samples if specified
    if only_botnet:
        idx = np.where(labels == 1)[0]
        samples = samples[idx]
        labels = labels[idx]
    print(f"[DEBUG] Filtered samples shape: {samples.shape}, labels shape: {labels.shape}")

    # One-hot encode labels
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)
    print(f"[DEBUG] Labels one-hot encoded shape: {labels_one_hot.shape}")

    # Load the model
    print(f"[DEBUG] Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("[DEBUG] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading model from {model_path}: {e}")
        return None, None

    # Initialize PGD attack
    perturbSamples = None
    success_rate = 0
    if method == "pgd":
        print(f"[DEBUG] clean_samples shape: {samples.shape}")
        attack_generator = PgdRandomRestart(
            model,
            eps=distance,
            alpha=1,
            num_iter=iterations,
            restarts=5,
            scaler=scaler,
            mins=min_features,
            maxs=max_features,
            mask_idx=mask_idx,
            eq_min_max=eq_min_max
        )
        print("[DEBUG] Running PGD attack...")
        try:
            perturbSamples = attack_generator.run_attack(samples, labels_one_hot)
            print("[DEBUG] PGD attack completed.")
        except Exception as e:
            print(f"[ERROR] Error during PGD attack: {e}")
            return None, None

    # Evaluate attack success rate
    if perturbSamples is not None:
        probas = np.squeeze(model.predict(perturbSamples))
        predictions = np.argmax(probas, axis=1)
        adv_success = predictions == 0  # Assuming class 0 is the adversarial target
        success_rate = np.sum(adv_success) / len(predictions) * 100
        print(f"Attack success rate: {success_rate:.2f}%")
    else:
        print("No perturbations generated.")

    return perturbSamples, success_rate

# Main execution
if __name__ == "__main__":
    print("Loading required files...")
    scaler = joblib.load('../data/neris/scaler.pkl')
    min_features, max_features = read_min_max('../data/neris/minimum.txt', '../data/neris/maximum.txt')
    mask_idx = np.load('../data/neris/mutable_idx.npy')
    eq_min_max = np.load('../data/neris/eq_min_max_idx.npy')

    model_path = '../out/neris/clean_10epochs/clean_model.keras'

    start_time = datetime.now()
    print("Starting attack...")

    try:
        perturbed_samples, success_rate = attack(
            method='pgd',
            model_path=model_path,
            samples_path='../data/neris/testing_samples.npy',
            labels_path='../data/neris/testing_labels.npy',
            distance=12,
            iterations=100,
            mask_idx=mask_idx,
            eq_min_max=eq_min_max
        )
        if perturbed_samples is not None:
            print(f"Final success rate: {success_rate:.2f}%")
    except Exception as e:
        print(f"[ERROR] Error during attack: {e}")

    end_time = datetime.now()
    print(f"Duration: {end_time - start_time}")