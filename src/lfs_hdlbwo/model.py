"""CBLSTM-AE forecasting network from workflow.md."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_cblstm_ae(
    sequence_length: int,
    num_features: int,
    conv_filters: int = 64,
    bilstm_units: int = 128,
    decoder_units: int = 64,
    learning_rate: float = 0.001,
    dense_units: int = 32,
    activation: str = "relu",
) -> Model:
    """Build the CBLSTM-AE architecture specified by FR-5."""

    inputs = Input(shape=(sequence_length, num_features), name="input")

    x = Conv1D(
        filters=conv_filters,
        kernel_size=3,
        padding="same",
        activation=activation,
        name="conv1",
    )(inputs)
    x = Conv1D(
        filters=conv_filters,
        kernel_size=3,
        padding="same",
        activation=activation,
        name="conv2",
    )(x)
    x = MaxPooling1D(pool_size=2, name="max_pool")(x)
    x = Bidirectional(
        LSTM(bilstm_units, return_sequences=True),
        name="bilstm_encoder",
    )(x)
    x = Flatten(name="flatten")(x)
    x = RepeatVector(sequence_length, name="repeat_vector")(x)
    x = LSTM(decoder_units, return_sequences=True, name="lstm_decoder")(x)
    x = TimeDistributed(
        Dense(dense_units, activation=activation),
        name="td_dense",
    )(x)
    outputs = TimeDistributed(Dense(1), name="forecast")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CBLSTM_AE")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def build_lfshdlbwo_forecaster(
    sequence_length: int,
    num_features: int,
    conv_filters: int = 64,
    bilstm_units: int = 128,
    decoder_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    activation: str = "relu",
) -> Model:
    """Build the legacy one-step forecaster kept for compatibility."""

    inputs = Input(shape=(sequence_length, num_features), name="input")
    x = Conv1D(
        filters=conv_filters,
        kernel_size=3,
        padding="same",
        activation=activation,
        name="conv1",
    )(inputs)
    x = Conv1D(
        filters=conv_filters,
        kernel_size=3,
        padding="same",
        activation=activation,
        name="conv2",
    )(x)
    x = MaxPooling1D(pool_size=2, name="max_pool")(x)
    x = Bidirectional(
        LSTM(bilstm_units, return_sequences=True),
        name="bilstm_encoder",
    )(x)
    x = LSTM(decoder_units, return_sequences=False, name="lstm_decoder")(x)
    x = Dropout(dropout, name="dropout")(x)
    x = Dense(dense_units, activation=activation, name="dense")(x)
    outputs = Dense(1, name="forecast")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LFS_HDLBWO_Forecaster")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model
