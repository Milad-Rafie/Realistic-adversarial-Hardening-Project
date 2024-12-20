import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from attack.fence.neris_attack_tf2 import Neris_attack
from attack.pgd.pgd_attack_art import PgdRandomRestart


def generate_adversarial_batch_fence(model, total, samples, labels, distance, iterations, scaler, mins, maxs, model_path):
    while True:
        # Save model
        model.save(model_path)

        # Initialize the attack generator
        attack_generator = Neris_attack(
            model_path=model_path, iterations=iterations, distance=distance, scaler=scaler, mins=mins, maxs=maxs
        )

        perturbSamples = []
        perturbLabels = []
        idxs = np.random.choice(range(0, len(samples)), size=total, replace=False)

        for i in idxs:
            sample = samples[i]
            sample = np.expand_dims(sample, axis=0)  # Ensure the shape is correct
            label = labels[i]

            # Generate an adversarial sample
            adversary = attack_generator.run_attack(sample, label)

            # Debug output for adversary
            print(f"[DEBUG] Adversary output: {adversary}")

            # Validate adversarial sample
            if adversary is not None and isinstance(adversary, np.ndarray) and adversary.size > 0:
                perturbSamples.append(adversary)
                perturbLabels.append(label)
            else:
                print(f"[DEBUG] Invalid adversary for index {i}. Skipping...")

        # Check if the batch has valid data
        if len(perturbSamples) == 0 or len(perturbLabels) == 0:
            print("[ERROR] No adversarial samples generated. Skipping this batch.")
            continue

        perturbSamples = np.array(perturbSamples)
        perturbLabels = np.array(perturbLabels)

        # Debug shapes
        print(f"[DEBUG] PerturbSamples shape: {perturbSamples.shape}")
        print(f"[DEBUG] PerturbLabels shape: {perturbLabels.shape}")

        yield perturbSamples, tf.keras.utils.to_categorical(perturbLabels, num_classes=2)
        

def generate_adversarial_batch_pgd(model, total, samples, labels, distance, iterations, scaler, mins, maxs, mutable_idx, eq_min_max_idx):
   
    attack_generator = PgdRandomRestart(
        model=model,
        eps=distance,
        alpha=1,
        num_iter=iterations,
        restarts=5,
        scaler=scaler,
        mins=mins,
        maxs=maxs,
        mutable_idx=mutable_idx,
        eq_min_max_idx=eq_min_max_idx
    )

    while True:
        idxs = np.random.choice(range(0, len(samples)), size=total, replace=False)
        batch_samples = samples[idxs]
        perturbLabels = labels[idxs]

        # Ensure labels are one-hot encoded to match the model's output
        if perturbLabels.ndim == 1:
            num_classes = model.output_shape[-1]  # Get the number of classes from the model output shape
            perturbLabels = tf.keras.utils.to_categorical(perturbLabels, num_classes=num_classes)

        perturbSamples = attack_generator.run_attack(batch_samples, np.argmax(perturbLabels, axis=1))

        # Debugging shapes
        print(f"[DEBUG] PerturbSamples shape: {perturbSamples.shape}")
        print(f"[DEBUG] PerturbLabels shape: {perturbLabels.shape}")

        yield np.array(perturbSamples), np.array(perturbLabels)
