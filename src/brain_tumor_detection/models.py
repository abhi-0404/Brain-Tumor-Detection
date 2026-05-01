"""Model builders for brain tumor MRI classification."""

from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adamax


def compile_model(model, learning_rate: float):
    """Compile a classification model with the project default loss and metric."""
    model.compile(
        optimizer=Adamax(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model(input_shape: tuple[int, int, int], class_count: int, learning_rate: float = 5e-4):
    """Build a compact CNN suitable as a first trainable baseline."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.1),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.15),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(192, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(0.35),
            Dense(class_count, activation="softmax"),
        ],
        name="brain_tumor_cnn",
    )
    return compile_model(model, learning_rate)


def build_mlp_model(input_shape: tuple[int, int, int], class_count: int, learning_rate: float = 4e-4):
    """Build a simple MLP baseline for comparison with convolutional models."""
    model = Sequential(
        [
            Flatten(input_shape=input_shape),
            Dense(384, activation="relu"),
            BatchNormalization(),
            Dropout(0.35),
            Dense(192, activation="relu"),
            Dropout(0.25),
            Dense(96, activation="relu"),
            Dropout(0.15),
            Dense(class_count, activation="softmax"),
        ],
        name="brain_tumor_mlp",
    )
    return compile_model(model, learning_rate)


def build_vgg16_model(input_shape: tuple[int, int, int], class_count: int, learning_rate: float = 2.5e-4):
    """Build a VGG16 transfer-learning model with a lightweight classification head."""
    vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    vgg_base.trainable = True
    for layer in vgg_base.layers[:-6]:
        layer.trainable = False

    model = Sequential(
        [
            vgg_base,
            GlobalAveragePooling2D(),
            Dense(320, activation="relu"),
            Dropout(0.45),
            Dense(class_count, activation="softmax"),
        ],
        name="brain_tumor_vgg16",
    )
    return compile_model(model, learning_rate)


MODEL_BUILDERS = {
    "cnn": build_cnn_model,
    "mlp": build_mlp_model,
    "vgg16": build_vgg16_model,
}

