import datetime
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_cloud as tfc
import tensorflow as tf
# import PIL
import os
import numpy as np
import argparse
import json

gcp_bucket = "picture-classifier24902"
job_labels = {"job": "mnist-example", "team": "keras-io", "user": "jonah"}


tfc.run(docker_image_bucket_name=gcp_bucket,
        job_labels=job_labels,
        )

# data_dir = pathlib.Path("C:/Users/Lotfi/git/picture_classifier/test_folder")
# data_dir = pathlib.Path("C:/Users/Lotfi/git/picture_classifier/pic")
# data_dir = os.path.join("gs://",gcp_bucket,"pic")
# data_dir = "./test_folder"
data_dir = pathlib.Path("./pic")
# data_dir = "https://console.cloud.google.com/storage/browser/picture-classifier24902/pic"
# data_dir = "gs://"+gcp_bucket+"/pic"
# data_dir = "gs://picture-classifier24902/training_job/save_at_1/assets"
batch_size = 32
img_height = 180
img_width = 180


def load_ds():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                   validation_split=0.2,
                                                                   subset="training",
                                                                   seed=123,
                                                                   image_size=(
                                                                       img_height, img_width),
                                                                   batch_size=batch_size,
                                                                   )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    return val_ds, train_ds


checkpoint_dir = "./training_checkpoints"


def make_local_callbacks():

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2, min_lr=0.000001
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )

    tensorboard_callback = (
        keras.callbacks.TensorBoard(
            log_dir="./logs", histogram_freq=1),
    )
    # ModelCheckpoint will save models after each epoch for retrieval later.
    json_log = open("loss_log.json", mode="wt", buffering=1)
    json_logging_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({"epoch": str(epoch), "loss": str(logs["lr"])}) + "\n"
        ),
        on_train_end=lambda logs: json_log.close(),
    )

    return [reduce_lr, json_logging_callback, checkpoint_callback, tensorboard_callback]


def make_callbacks():

    checkpoint_path = os.path.join(
        "gs://", gcp_bucket, "training_job", "save_at_{epoch}"
    )

    tensorboard_path = os.path.join(  # Timestamp included to enable timeseries graphs
        "gs://", gcp_bucket, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2, min_lr=0.000001
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )

    # TensorBoard will store logs for each epoch and graph performance for us.
    tensorboard_callback = (
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_path, histogram_freq=1),
    )
    # ModelCheckpoint will save models after each epoch for retrieval later.
    json_log = open("loss_log.json", mode="wt", buffering=1)
    json_logging_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({"epoch": str(epoch), "loss": str(logs["lr"])}) + "\n"
        ),
        on_train_end=lambda logs: json_log.close(),
    )

    return [reduce_lr, json_logging_callback, checkpoint_callback, tensorboard_callback]


def make_model():
    num_classes = 43
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip(
                "horizontal", input_shape=(img_height, img_width, 3)
            ),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    model = Sequential(
        [
            data_augmentation,
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def train():
    # physical_devices = tf.config.experimental.list_physical_devices("GPU")
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(
    #     physical_devices[0], True)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    val_ds, train_ds = load_ds()

    print(type(train_ds))

    normalization_layer = layers.experimental.preprocessing.Rescaling(
        1.0 / 255)

    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    model = make_model()

    # model = tf.keras.models.load_model(checkpoint_dir)

    epochs = 1
    if tfc.remote():
        epochs = 20
        callbacks = make_callbacks()
        print("yessir")
    else:
        epochs = 20
        callbacks = make_local_callbacks()
        print("nosir")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        initial_epoch=0,
        callbacks=callbacks,
        shuffle=True,
    )

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label="Training Accuracy")
    # plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    # plt.legend(loc="lower right")
    # plt.title("Training and Validation Accuracy")

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label="Training Loss")
    # plt.plot(epochs_range, val_loss, label="Validation Loss")
    # plt.legend(loc="upper right")
    # plt.title("Training and Validation Loss")
    # plt.show()


def chat():
    x = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices={"train", "chat", "bmark"},
        default="train",
        help="mode. if not specified, it's in the train mode",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "chat":
        chat()


if __name__ == "__main__":
    main()
