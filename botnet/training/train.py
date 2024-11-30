import numpy as np
import tensorflow as tf
import json
from helpers import create_DNN, get_train_data, get_model_data, get_processing_data

def train(config, method="clean", distance=12, save_data=True):
    # Load training data
    x_train, y_train, x_test, y_test = get_train_data(config)
    LAYERS, INPUT_DIM, LR = get_model_data(config)
    scaler, min_features, max_features = get_processing_data(config)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
    epochs = config["epochs"]

    for lrate in LR:
        # Create the model
        model = create_DNN(units=LAYERS, input_dim_param=INPUT_DIM, lr_param=lrate)
        print("Training model...")

        # Train the model
        history_obj = model.fit(
            x_train,
            y_train,
            verbose=1,
            epochs=epochs,
            batch_size=64,
            shuffle=True
        )

        if save_data:
            save_model_and_history(config, model, history_obj)

def save_model_and_history(config, model, history_obj):
    """
    Save the trained model in multiple formats and the training history.
    """
    # Save the model in .keras format
    keras_model_path = config["path_to_save"] + "/clean_model.keras"
    print(f"Saving model in .keras format to {keras_model_path}...")
    model.save(keras_model_path)

    # Save the model in .h5 format
    h5_model_path = config["path_to_save"] + "/clean_model.h5"
    print(f"Saving model in .h5 format to {h5_model_path}...")
    model.save(h5_model_path)

    # Save the model in SavedModel format
    saved_model_dir = config["path_to_save"] + "/clean_model_saved"
    print(f"Exporting model in TensorFlow SavedModel format to {saved_model_dir}...")
    tf.saved_model.save(model, saved_model_dir)  # Correct method for saving in SavedModel format

    # Save the training history
    history_path = config["path_to_save"] + "/clean_model_history.npy"
    print(f"Saving training history to {history_path}...")
    with open(history_path, "wb") as f:
        np.save(f, dict(history_obj.history))  # Ensure it's a dictionary

    print(f"Model and history saved successfully at {config['path_to_save']}")

if __name__ == "__main__":
    # Load configuration file
    config_file = "config/neris.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    # Train the model
    train(config)