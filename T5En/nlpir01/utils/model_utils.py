import tensorflow as tf
import numpy as np
import pandas as pd

from format_checker.main import check_format
from scorer.main import evaluate
from os.path import join
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.text_sequencer import Sequencer
from utils.rid_counts_extractor import RIDCountsExtractor
from utils.path_utils import get_dataframe_from_file, get_dataframe_from_file_list
from utils.text_features import get_feature_matrix, get_pos_df
from utils.global_parameters import COL_NAMES, TEXT_COLUMN, LABEL_COLUMN, BATCH_SIZE, EPOCHS
from utils.global_parameters import NUM_WORDS, SEQ_LEN, EMBEDDING_SIZE


def run_model(model, x_train, y_train, train_df, test_debates, test_path, results_path, test1_df, prediction_prefix,
              batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)
    history_dict = history.history

    # test1_df only to check class prediction accuracy
    _, _, x_test1, y_test1, _ = get_sequences_from_dataset(train_df, test1_df)
    loss, accuracy = model.evaluate(x_test1, y_test1, verbose=0)
    raw_predictions = model.predict(x_test1)
    class_predictions = [class_of(x) for x in raw_predictions]

    # ranking predictions on all debates
    for test_file_name in test_debates:
        test_df = get_dataframe_from_file(join(test_path, test_file_name), COL_NAMES)

        _, _, x_test, y_test, _ = get_sequences_from_dataset(train_df, test_df)
        _, _ = model.evaluate(x_test, y_test, verbose=verbose)
        predictions = model.predict(x_test).T[0]

        with open(join(results_path, prediction_prefix + test_file_name), "w") as results_file:
            for line_num, dist in zip(test_df["line_number"], predictions):
                results_file.write("{}\t{:.20f}\n".format(line_num, dist))

    return loss, accuracy, class_predictions, y_test1, history_dict


def fit_model(model, x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)
    history_dict = history.history
    return model, history_dict


def predict_model(model, train_df, test_debates, test_path, results_path, prediction_prefix):
    # ranking predictions on all debates
    for test_file_name in test_debates:
        test_df = get_dataframe_from_file(join(test_path, test_file_name), COL_NAMES)

        _, _, x_test, _, _ = get_sequences_from_dataset(train_df, test_df)
        predictions = model.predict(x_test).T[0]
        with open(join(results_path, prediction_prefix + test_file_name), "w") as results_file:
            for line_num, dist in zip(test_df["line_number"], predictions):
                results_file.write("{}\t{:.20f}\n".format(line_num, dist))


def evaluate_model(model, train_df, test1_df, verbose=0):
    # test1_df only to check class prediction accuracy
    _, _, x_test1, y_test1, _ = get_sequences_from_dataset(train_df, test1_df)
    loss, accuracy = model.evaluate(x_test1, y_test1, verbose=0)
    raw_predictions = model.predict(x_test1)
    class_predictions = [class_of(x) for x in raw_predictions]

    return loss, accuracy, class_predictions, y_test1


def run_model_rid(model, x_train_rid, y_train, train_df, test_debates, test_path, results_path, test1_df, prediction_prefix,
                  batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):

    history = model.fit(x_train_rid, y_train, batch_size=batch_size,
                        epochs=epochs, validation_split=0.2, verbose=verbose)
    history_dict = history.history

    # test1_df only to check class prediction accuracy
    _, _, _, y_test1, _ = get_sequences_from_dataset(train_df, test1_df)
    _, x_test_rid1 = get_rid_counts_from_dataset(train_df, test1_df)
    loss, accuracy = model.evaluate(x_test_rid1, y_test1, verbose=verbose)
    raw_predictions = model.predict(x_test_rid1)
    class_predictions = [class_of(x) for x in raw_predictions]

    # ranking predictions on all debates
    for test_file_name in test_debates:
        test_df = get_dataframe_from_file(join(test_path, test_file_name), COL_NAMES)

        _, _, _, y_test, _ = get_sequences_from_dataset(train_df, test_df)
        _, x_test_rid = get_rid_counts_from_dataset(train_df, test_df)
        _, _ = model.evaluate(x_test_rid, y_test, verbose=0)
        predictions = model.predict(x_test_rid).T[0]

        with open(join(results_path, prediction_prefix + test_file_name), "w") as results_file:
            for line_num, dist in zip(test_df["line_number"], predictions):
                results_file.write("{}\t{:.20f}\n".format(line_num, dist))

    return loss, accuracy, class_predictions, y_test1, history_dict


def predict_model_rid(model, train_df, test_debates, test_path, results_path, prediction_prefix):
    # ranking predictions on all debates
    for test_file_name in test_debates:
        test_df = get_dataframe_from_file(join(test_path, test_file_name), COL_NAMES)

        _, _, _, _, _ = get_sequences_from_dataset(train_df, test_df)
        _, x_test_rid = get_rid_counts_from_dataset(train_df, test_df)
        predictions = model.predict(x_test_rid).T[0]
        with open(join(results_path, prediction_prefix + test_file_name), "w") as results_file:
            for line_num, dist in zip(test_df["line_number"], predictions):
                results_file.write("{}\t{:.20f}\n".format(line_num, dist))


def evaluate_model_rid(model, train_df, test1_df, verbose=0):
    # test1_df only to check class prediction accuracy
    _, _, _, y_test1, _ = get_sequences_from_dataset(train_df, test1_df)
    _, x_test_rid1 = get_rid_counts_from_dataset(train_df, test1_df)
    loss, accuracy = model.evaluate(x_test_rid1, y_test1, verbose=verbose)
    raw_predictions = model.predict(x_test_rid1)
    class_predictions = [class_of(x) for x in raw_predictions]

    return loss, accuracy, class_predictions, y_test1


def run_model_text(model, x_train_text, y_train, train_df, test_debates, test_path, results_path, test1_df,
                   prediction_prefix, train_tiv_fit, train_pos_tiv_fit,
                   batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):

    history = model.fit(x_train_text, y_train, batch_size=batch_size,
                        epochs=epochs, validation_split=0.2, verbose=verbose)
    history_dict = history.history

    # test1_df only to check class prediction accuracy
    _, _, _, y_test1, _ = get_sequences_from_dataset(train_df, test1_df)
    x_test_text1 = get_test_text_features(test1_df, train_tiv_fit, train_pos_tiv_fit)
    loss, accuracy = model.evaluate(x_test_text1, y_test1, verbose=verbose)
    raw_predictions = model.predict(x_test_text1)
    class_predictions = [class_of(x) for x in raw_predictions]

    # ranking predictions on all debates
    for test_file_name in test_debates:
        test_df = get_dataframe_from_file(join(test_path, test_file_name), COL_NAMES)

        _, _, _, y_test, _ = get_sequences_from_dataset(train_df, test_df)
        x_test_text = get_test_text_features(test_df, train_tiv_fit, train_pos_tiv_fit)
        _, _ = model.evaluate(x_test_text, y_test, verbose=0)
        predictions = model.predict(x_test_text).T[0]

        with open(join(results_path, prediction_prefix + test_file_name), "w") as results_file:
            for line_num, dist in zip(test_df["line_number"], predictions):
                results_file.write("{}\t{:.20f}\n".format(line_num, dist))

    return loss, accuracy, class_predictions, y_test1, history_dict


def predict_model_text(model, train_df, test_debates, test_path, results_path, prediction_prefix,
                       train_tiv_fit, train_pos_tiv_fit):
    # ranking predictions on all debates
    for test_file_name in test_debates:
        test_df = get_dataframe_from_file(join(test_path, test_file_name), COL_NAMES)

        _, _, _, _, _ = get_sequences_from_dataset(train_df, test_df)
        x_test_text = get_test_text_features(test_df, train_tiv_fit, train_pos_tiv_fit)
        predictions = model.predict(x_test_text).T[0]
        with open(join(results_path, prediction_prefix + test_file_name), "w") as results_file:
            for line_num, dist in zip(test_df["line_number"], predictions):
                results_file.write("{}\t{:.20f}\n".format(line_num, dist))


def evaluate_model_text(model, train_df, test1_df, train_tiv_fit, train_pos_tiv_fit, verbose=0):
    # test1_df only to check class prediction accuracy
    _, _, _, y_test1, _ = get_sequences_from_dataset(train_df, test1_df)
    x_test_text1 = get_test_text_features(test1_df, train_tiv_fit, train_pos_tiv_fit)
    loss, accuracy = model.evaluate(x_test_text1, y_test1, verbose=verbose)
    raw_predictions = model.predict(x_test_text1)
    class_predictions = [class_of(x) for x in raw_predictions]

    return loss, accuracy, class_predictions, y_test1


def get_sequences_from_dataset(train_df, test_df, sequecer_option="keras"):
    if sequecer_option == "keras":
        return get_sequences_from_dataset_keras(train_df, test_df)
    else:
        return get_sequences_from_dataset_custom(train_df, test_df)


def get_sequences_from_dataset_custom(train_df, test_df):
    sequencer = Sequencer(train_df[TEXT_COLUMN], NUM_WORDS, tokenizer="stanfordnlp", lang="en")
    word_index = sequencer.word_index

    train_seqs = sequencer.fit_on_text(train_df[TEXT_COLUMN], SEQ_LEN)
    test_seqs = sequencer.fit_on_text(test_df[TEXT_COLUMN], SEQ_LEN)

    return train_seqs, train_df[LABEL_COLUMN].values, test_seqs, test_df[LABEL_COLUMN].values, word_index


def get_sequences_from_dataset_keras(train_df, test_df):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=NUM_WORDS, oov_token="<UNK>")
    tokenizer.fit_on_texts(train_df[TEXT_COLUMN])

    train_seqs = tokenizer.texts_to_sequences(train_df[TEXT_COLUMN])
    test_seqs = tokenizer.texts_to_sequences(test_df[TEXT_COLUMN])

    word_index = tokenizer.word_index

    train_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, maxlen=SEQ_LEN, padding="post")
    test_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        test_seqs, maxlen=SEQ_LEN, padding="post")

    return train_seqs, train_df[LABEL_COLUMN].values, test_seqs, test_df[LABEL_COLUMN].values, word_index


def get_rid_counts_from_dataset(train_df, test_df):
    import os
    sequencer = Sequencer(train_df[TEXT_COLUMN], NUM_WORDS)

    train_tokens = sequencer.get_words(train_df[TEXT_COLUMN])
    test_tokens = sequencer.get_words(test_df[TEXT_COLUMN])

    rid_count_extractor = RIDCountsExtractor(
        "/Users/icobx/Documents/skola/dp/code/NLP-IR-UNED-at-CheckThat-2020/T5En/nlpir01/resources/rid_en.dic"
    )
    _, _, train_rid_percentages = rid_count_extractor.get_rid_counts_list(train_tokens)
    _, _, test_rid_percentages = rid_count_extractor.get_rid_counts_list(test_tokens)

    return train_rid_percentages, test_rid_percentages


def get_train_text_features(train_df, oversampling=0):

    count_class_0, count_class_1 = train_df.label.value_counts()
    print("Class 0 count: ", count_class_0, ", class 1 count:", count_class_1)
    if oversampling == 1:
        # --- random oversampling class 1 ---
        class_0_df = train_df[train_df[LABEL_COLUMN] == 0]
        class_1_df = train_df[train_df[LABEL_COLUMN] == 1]
        class_1_over_df = class_1_df.sample(count_class_0, replace=True)
        train_df = pd.concat([class_0_df, class_1_over_df], axis=0)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        count_class_0, count_class_1 = train_df.label.value_counts()
        print("After resampling, class 0 count: ", count_class_0, ", class 1 count:", count_class_1)

    # --- tf-idf features ---
    tiv = TfidfVectorizer(ngram_range=(1, 1))
    train_tiv_fit = tiv.fit(train_df[TEXT_COLUMN])
    train_tiv_fit_transform = train_tiv_fit.transform(train_df[TEXT_COLUMN])
    train_1_x = train_tiv_fit_transform.toarray()

    # --- tf-idf pos features ---
    train_pos_df = get_pos_df(train_df)
    pos_tiv = TfidfVectorizer(ngram_range=(1, 1))
    train_pos_tiv_fit = pos_tiv.fit(train_pos_df["text_pos"])
    train_2_x = train_pos_tiv_fit.transform(train_pos_df["text_pos"]).toarray()

    # --- attribute-based language features ---
    train_3_x = get_feature_matrix(train_df[TEXT_COLUMN], train_1_x.shape[0]).toarray()

    # --- concatenate matrix ---
    train_x = np.hstack((train_1_x, train_2_x, train_3_x))

    return train_x, train_tiv_fit, train_pos_tiv_fit


def get_test_text_features(test_df, train_tiv_fit, train_pos_tiv_fit):

    test_1_x = train_tiv_fit.transform(test_df["text"]).toarray()

    # --- tf-idf pos features ---
    test_pos_df = get_pos_df(test_df)
    test_2_x = train_pos_tiv_fit.transform(test_pos_df["text_pos"]).toarray()

    # --- attribute-based language features ---
    test_3_x = get_feature_matrix(test_df["text"], test_1_x.shape[0]).toarray()

    # --- concatenate matrix ---
    test_x = np.hstack((test_1_x, test_2_x, test_3_x))

    return test_x


def get_embedding_layer(pretrained_embedding_path, word_index):
    embeddings_index = {}
    embeddings_file = open(pretrained_embedding_path, encoding="utf-8")
    for line in embeddings_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    embeddings_file.close()

    embedding_matrix = np.zeros((len(word_index) + 1, coefs.shape[0]))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def class_of(binary_raw_value):
    if binary_raw_value < 0.5:
        return 0
    else:
        return 1


def check_and_evaluate(test_debates, test_path, results_path, prediction_prefix):
    total_avg_precision = 0
    for test_file_name in test_debates:
        test_file_name_with_path = join(test_path, test_file_name)
        result_file_name_with_path = join(results_path, prediction_prefix + test_file_name)

        if check_format(result_file_name_with_path):
            _, _, avg_precision, _, _ = evaluate(test_file_name_with_path, result_file_name_with_path)
            total_avg_precision = total_avg_precision + avg_precision

    med_av_precision = total_avg_precision / len(test_debates)
    return med_av_precision
