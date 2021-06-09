import time
from pathlib import Path

import tensorflow as tf

import config


def setup_model():
    """
    Setting up the base model and compling it.

    Returns:
        tfModel: complied TF model to be trained.
    """
    # initial architecture
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(128,)), tf.keras.layers.Dense(2),]
    )
    # compiling the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(config.OPTIMIZER_RATE),
        metrics=config.METRICS,
    )
    return model


def setup_callbacks():
    """
    Setting up all required callbacks for the model training.

    Returns:
        [List]: Lit of callbacks to used during model training.
    """
    # csv logging
    run_id = time.strftime(f"run_{config.MODEL_NAME}_%d_%m_%Y-%H_%M")
    csv_path = Path(Path.cwd(), config.PATH_LOGS)
    csv_path.mkdir(parents=True, exist_ok=True)
    csv_logger = tf.keras.callbacks.CSVLogger(str(csv_path) + f"/{run_id}.csv", append=True, separator=",")
    return [csv_logger]


def plot_model_architecture(model):
    """
    ...
    """
    return None
