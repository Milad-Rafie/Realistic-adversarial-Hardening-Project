import numpy as np
import tensorflow as tf
from art.attacks.evasion import ProjectedGradientDescent as PGD
from art.estimators.classification.tensorflow import TensorFlowV2Classifier as kc

class PgdRandomRestart:
    def __init__(self, model, eps, alpha, num_iter, restarts, scaler, mins, maxs, mutable_idx, eq_min_max_idx):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.num_iter = num_iter
        self.restarts = restarts
        self.scaler = scaler
        self.clip_min = self.scaler.transform(np.array(mins).reshape(1, -1))
        self.clip_max = self.scaler.transform(np.array(maxs).reshape(1, -1))
        self.clip_max[0][eq_min_max_idx] += 1e-9
        self.mutable_idx = mutable_idx
        self.eq_min_max_idx = eq_min_max_idx

    def run_attack(self, clean_samples, true_labels):
        print(f"[DEBUG] clean_samples shape: {clean_samples.shape}")
        print(f"[DEBUG] true_labels shape: {true_labels.shape}")

        # ART classifier
        kc_classifier = kc(
            self.model,
            clip_values=(self.clip_min, self.clip_max),
            nb_classes=2,
            input_shape=(clean_samples.shape[1],),
            loss_object=tf.keras.losses.BinaryCrossentropy()
        )

        # PGD attack initialization
        pgd = PGD(kc_classifier)
        pgd.set_params(
            eps=self.eps,
            verbose=False,
            max_iter=self.num_iter,
            num_random_init=self.restarts,
            norm=2,
            eps_step=self.alpha,
            targeted=True
        )

        # Run the attack
        return pgd.generate(x=clean_samples, y=true_labels)
