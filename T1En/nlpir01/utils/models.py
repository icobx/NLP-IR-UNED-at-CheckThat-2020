from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, LSTM, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import Model

from utils.global_parameters import EMBEDDING_SIZE, NUM_WORDS


def get_ffnn_emb_model(
    embedding_matrix,
    word_index,
    input_length,
    hidden_layer_size,
    activation="sigmoid",
    optimizer="adam",
    verbose=0
):
    sequence_input = Input(shape=(input_length,), dtype="int32")
    if embedding_matrix is None:
        embedded_sequences = Embedding(
            NUM_WORDS, EMBEDDING_SIZE)(sequence_input)
    else:
        embedded_sequences = Embedding(
            len(word_index) + 1,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False,
        )(sequence_input)

    x = GlobalAveragePooling1D()(embedded_sequences)

    if hidden_layer_size > 0:
        x = Dense(hidden_layer_size, activation=activation)(x)

    preds = Dense(1, activation="sigmoid")(x)
    model = Model(sequence_input, preds)

    if verbose == 1:
        model.summary()
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_cnn_emb_model(embedding_matrix, word_index, input_length, hidden_layer_size,
                      activation="sigmoid", optimizer="adam", verbose=0):
    sequence_input = Input(shape=(input_length,), dtype="int32")
    if embedding_matrix is None:
        embedded_sequences = Embedding(
            NUM_WORDS, EMBEDDING_SIZE)(sequence_input)
    else:
        embedded_sequences = Embedding(
            len(word_index) + 1,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False)(sequence_input)
    x = Conv1D(hidden_layer_size, 5, activation=activation)(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(hidden_layer_size, 5, activation=activation)(x)
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    x = Dense(hidden_layer_size, activation=activation)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(sequence_input, preds)

    if verbose == 1:
        model.summary()
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_lstm_emb_model(embedding_matrix, word_index, input_length, hidden_layer_size=EMBEDDING_SIZE,
                       dropout=0.2, recurrent_dropout=0.2, optimizer="adam", verbose=0):
    sequence_input = Input(shape=(input_length,), dtype="int32")
    if embedding_matrix is None:
        embedded_sequences = Embedding(
            NUM_WORDS, EMBEDDING_SIZE)(sequence_input)
    else:
        embedded_sequences = Embedding(
            len(word_index) + 1,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False)(sequence_input)
    x = LSTM(
        hidden_layer_size,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout)(embedded_sequences)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(sequence_input, preds)

    if verbose == 1:
        model.summary()
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_bilstm_emb_model(embedding_matrix, word_index, input_length, hidden_layer_size=EMBEDDING_SIZE,
                         dropout=0.2, recurrent_dropout=0, activation="tanh", optimizer="adam", verbose=0):
    sequence_input = Input(shape=(input_length,), dtype="int32")
    if embedding_matrix is None:
        embedded_sequences = Embedding(
            NUM_WORDS, EMBEDDING_SIZE)(sequence_input)
    else:
        embedded_sequences = Embedding(
            len(word_index) + 1,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False)(sequence_input)
    x = Bidirectional(LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        activation=activation))(embedded_sequences)
    x = Bidirectional(LSTM(
        hidden_layer_size,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        activation=activation))(x)
    x = Dense(hidden_layer_size, activation=activation)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(sequence_input, preds)

    if verbose == 1:
        model.summary()
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_ffnn_model(input_length, hidden_layer_size,
                   dropout=0.4, activation="sigmoid", optimizer="adam", verbose=0):
    sequence_input = Input(shape=(input_length,), dtype="float32")
    x = Dense(hidden_layer_size, activation=activation)(sequence_input)
    x = BatchNormalization()(x)
    x = Dense(int(hidden_layer_size / 2), activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dense(int(hidden_layer_size / 4), activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(sequence_input, preds)

    if verbose == 1:
        model.summary()
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model
