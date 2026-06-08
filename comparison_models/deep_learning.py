"""Neural baseline model builders."""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GRU,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from lfs_hdlbwo.model import build_cblstm_ae

logger = logging.getLogger(__name__)


def build_lstm_baseline(
    sequence_length: int,
    num_features: int,
    units: int = 64,
    dropout: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> Model:
    """Build a one-step LSTM forecasting baseline."""

    model = Sequential(
        [
            Input(shape=(sequence_length, num_features)),
            LSTM(units),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ],
        name="LSTM_Baseline",
    )
    return _compile(model, learning_rate)


def build_bilstm_baseline(
    sequence_length: int,
    num_features: int,
    units: int = 64,
    dropout: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> Model:
    """Build a one-step bidirectional LSTM forecasting baseline."""

    model = Sequential(
        [
            Input(shape=(sequence_length, num_features)),
            Bidirectional(LSTM(units)),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ],
        name="BiLSTM_Baseline",
    )
    return _compile(model, learning_rate)


def build_gru_baseline(
    sequence_length: int,
    num_features: int,
    units: int = 64,
    dropout: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> Model:
    """Build a one-step GRU forecasting baseline."""

    model = Sequential(
        [
            Input(shape=(sequence_length, num_features)),
            GRU(units),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ],
        name="GRU_Baseline",
    )
    return _compile(model, learning_rate)


def build_cnn_lstm_baseline(
    sequence_length: int,
    num_features: int,
    conv_filters: int = 64,
    lstm_units: int = 64,
    dropout: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> Model:
    """Build a one-step CNN-LSTM forecasting baseline."""

    inputs = Input(shape=(sequence_length, num_features))
    x = Conv1D(conv_filters, kernel_size=3, padding="same", activation="relu")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(dropout)(x)
    x = Dense(dense_units, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name="CNN_LSTM_Baseline")
    return _compile(model, learning_rate)


def build_cnn_bilstm_baseline(
    sequence_length: int,
    num_features: int,
    conv_filters: int = 64,
    bilstm_units: int = 64,
    dropout: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> Model:
    """Build a one-step CNN-BiLSTM forecasting baseline."""

    inputs = Input(shape=(sequence_length, num_features))
    x = Conv1D(conv_filters, kernel_size=3, padding="same", activation="relu")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(bilstm_units))(x)
    x = Dropout(dropout)(x)
    x = Dense(dense_units, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name="CNN_BiLSTM_Baseline")
    return _compile(model, learning_rate)


def build_cblstm_ae_baseline(
    sequence_length: int,
    num_features: int,
    conv_filters: int = 64,
    bilstm_units: int = 128,
    decoder_units: int = 64,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> Model:
    """Build the plain CBLSTM-AE baseline from workflow.md."""

    return build_cblstm_ae(
        sequence_length=sequence_length,
        num_features=num_features,
        conv_filters=conv_filters,
        bilstm_units=bilstm_units,
        decoder_units=decoder_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
    )


def _compile(model: Model, learning_rate: float) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    logger.debug("Built model %s output_shape=%s", model.name, model.output_shape)
    return model
