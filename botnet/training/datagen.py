import sys
sys.path.append('..')

import numpy as np
from attack.pgd.pgd_attack_art import PgdRandomRestart

def generate_adversarial_batch_pgd(
    model,
    total,
    samples,
    labels,
    distance,
    iterations,
    scaler,
    mins,
    maxs,
    mask_idx,
    eq_min_max
):
    """
    Yields (adv_samples, train_labels_onehot) indefinitely.
    - adv_samples: (batch, features)
    - train_labels_onehot: (batch, 2)
    """
    attack_generator = PgdRandomRestart(
        model=model,
        eps=distance,
        alpha=1,
        num_iter=iterations,
        restarts=3,
        scaler=scaler,
        mins=mins,
        maxs=maxs,
        mutable_idx=mask_idx,
        eq_min_max_idx=eq_min_max
    )

    n_samples = samples.shape[0]
    while True:
        idxs = np.random.choice(n_samples, size=total, replace=False)
        batch_X = samples[idxs]
        batch_Y_onehot = labels[idxs]

        # Convert one-hot to ints for attack
        batch_Y_int = np.argmax(batch_Y_onehot, axis=1).reshape(-1, 1)

        adv_X = attack_generator.run_attack(batch_X, batch_Y_int)

        yield adv_X, batch_Y_onehot
