import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification.tensorflow import TensorFlowV2Classifier

class PgdRandomRestart:
    def __init__(
        self,
        model,
        eps,
        alpha,
        num_iter,
        restarts,
        scaler,
        mins,
        maxs,
        mutable_idx,
        eq_min_max_idx
    ):
        # Determine how many logits your base model already spits out:
        base_out_dim = model.output_shape[-1]

        if base_out_dim == 1:
            # If it's a single-sigmoid, wrap into two‐class [1−p, p]:
            input_dim = model.input_shape[-1]
            inp = Input(shape=(input_dim,))
            p1  = model(inp)                      # (batch,1)
            p0  = Lambda(lambda x: 1.0 - x)(p1)   # (batch,1)
            out = Concatenate(axis=1)([p0, p1])   # (batch,2)
            self.model = Model(inputs=inp, outputs=out)
            self.nb_classes = 2
        else:
            # If it's already multi‐class, use it directly:
            self.model = model
            self.nb_classes = base_out_dim

        # Store attack params
        self.eps  = eps
        self.alpha = alpha
        self.num_iter  = num_iter
        self.restarts  = restarts
        self.mutable_idx    = mutable_idx
        self.eq_min_max_idx = eq_min_max_idx

        # Precompute clipping bounds
        mins_arr = np.array(mins).reshape(1, -1)
        maxs_arr = np.array(maxs).reshape(1, -1)
        self.clip_min = scaler.transform(mins_arr)
        self.clip_max = scaler.transform(maxs_arr)
        self.clip_max[0, eq_min_max_idx] += 1e-9  # avoid strict equality


    def run_attack(self, X_clean, y_true):
        # y_true: (batch,1) integers {0,1}
        y_int = y_true.reshape(-1)

        # Build target for a *targeted* flip: flip 0↔1
        y_target_class = y_int ^ 1  # bitwise XOR flips 0↔1
        # Convert to one-hot matching nb_classes:
        y_target = tf.keras.utils.to_categorical(y_target_class, num_classes=self.nb_classes)

        # Mask features:
        mask = np.zeros(X_clean.shape[1], dtype=float)
        mask[self.mutable_idx] = 1.0

        # Wrap in ART classifier:
        classifier = TensorFlowV2Classifier(
            model=self.model,
            loss_object=CategoricalCrossentropy(),
            nb_classes=self.nb_classes,
            input_shape=(X_clean.shape[1],),
            clip_values=(self.clip_min, self.clip_max),
        )

        # Configure PGD
        pgd = ProjectedGradientDescent(estimator=classifier)
        pgd.set_params(
            eps=self.eps,
            eps_step=self.alpha,
            max_iter=self.num_iter,
            num_random_init=self.restarts,
            targeted=True,
            norm=2,
            verbose=False,
        )

        # Generate and return adversarial samples:
        return pgd.generate(x=X_clean, y=y_target, mask=mask)

# Not intended to run standalone
if __name__ == "__main__":
    print("This module provides the PgdRandomRestart class.")
