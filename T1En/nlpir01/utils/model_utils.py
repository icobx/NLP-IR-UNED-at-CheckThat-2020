import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from os.path import join
from os import path

from utils.tweets_graph import TweetsGraph
from utils.text_sequencer import Sequencer
from utils.global_parameters import RESOURCES_PATH, RESULTS_PATH, SEQ_LEN, TEXT_COLUMN
from utils.global_parameters import LABEL_COLUMN, RESULTS_PER_CLAIM, NUM_WORDS
from utils.global_parameters import EMBEDDING_SIZE, BATCH_SIZE, EPOCHS, COL_NAMES


def run_model_binary(model, x_train, y_train, x_test, y_test,
                     batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):

    history = model.fit(x_train, y_train, batch_size, epochs,
                        validation_split=0.2, verbose=verbose)

    if y_test is not None:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    else:
        loss = None
        accuracy = None

    raw_predictions = model.predict(x_test)
    class_predictions = [class_of(x) for x in raw_predictions]
    history_dict = history.history

    return loss, accuracy, class_predictions, history_dict, raw_predictions


def fit_model_binary(model, x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0):
    history = model.fit(x_train, y_train, batch_size, epochs,
                        validation_split=0.2, verbose=verbose)
    history_dict = history.history
    return model, history_dict


def evaluate_model_binary(model, x_test, y_test, verbose=0):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    return loss, accuracy


def predict_model_binary(model, x_test):
    raw_predictions = model.predict(x_test)
    class_predictions = [class_of(x) for x in raw_predictions]
    return class_predictions, raw_predictions


def class_of(binary_raw_value):
    if binary_raw_value < 0.5:
        return 0
    else:
        return 1


def export_results(test_df, predictions, results_path, results_file_prefix, run_id):
    test_df["prediction"] = predictions
    if 'topic_id' not in test_df.columns:
        test_df['topic_id'] = 'political_debates'

    if test_df.loc[0, 'topic_id'] == 'covid-19':
        test_df.rename(columns={'tweet_id': 'id'}, inplace=True)

    results_df = test_df.sort_values(
        by=["topic_id", "prediction"], ascending=False)
    with open(join(results_path, results_file_prefix + run_id), "w") as results_file:
        last_topic = ""
        for _, row in results_df.iterrows():
            if last_topic != row["topic_id"]:
                rank = 1

            if rank <= RESULTS_PER_CLAIM or RESULTS_PER_CLAIM == 0:
                results_file.write("{}\t{}\t{:.15f}\t{}\n".format(row["topic_id"], row["id"],
                                                                  row["prediction"], run_id))
            rank = rank + 1
            last_topic = row["topic_id"]


def get_sequences_from_dataset(dataset, train_df, test_df, clef_submission=0):
    text_col, label_col = COL_NAMES[dataset].values()

    sequencer = Sequencer(train_df[text_col],
                          NUM_WORDS, tokenizer="nltk", lang="en")
    word_index = sequencer.word_index

    print(f"Found {sequencer.unique_word_count} unique tokens.")

    train_seqs = sequencer.fit_on_text(train_df[text_col], SEQ_LEN)
    test_seqs = sequencer.fit_on_text(test_df[text_col], SEQ_LEN)

    if clef_submission == 1:
        return train_seqs, train_df[label_col].values, test_seqs, None, word_index
    else:
        return train_seqs, train_df[label_col].values, test_seqs, test_df[label_col].values, word_index


def get_sequences_from_dataset_gf(train_df, test_df, clef_submission=0):
    sequencer = Sequencer(train_df[TEXT_COLUMN],
                          NUM_WORDS, tokenizer="nltk", lang="en")
    word_index = sequencer.word_index

    print(f"Found {sequencer.unique_word_count} unique tokens.")

    train_0_seqs = sequencer.fit_on_text(train_df[TEXT_COLUMN], SEQ_LEN)
    test_0_seqs = sequencer.fit_on_text(test_df[TEXT_COLUMN], SEQ_LEN)

    rel_train_1_seqs = sequencer.fit_on_text(train_df["rel_text_1"], SEQ_LEN)
    rel_test_1_seqs = sequencer.fit_on_text(test_df["rel_text_1"], SEQ_LEN)
    rel_train_2_seqs = sequencer.fit_on_text(train_df["rel_text_2"], SEQ_LEN)
    rel_test_2_seqs = sequencer.fit_on_text(test_df["rel_text_2"], SEQ_LEN)
    rel_train_3_seqs = sequencer.fit_on_text(train_df["rel_text_3"], SEQ_LEN)
    rel_test_3_seqs = sequencer.fit_on_text(test_df["rel_text_3"], SEQ_LEN)

    train_seqs = np.concatenate(
        (train_0_seqs, rel_train_1_seqs, rel_train_2_seqs, rel_train_3_seqs), axis=1)
    test_seqs = np.concatenate(
        (test_0_seqs, rel_test_1_seqs, rel_test_2_seqs, rel_test_3_seqs), axis=1)

    if clef_submission == 1:
        return train_seqs, train_df[LABEL_COLUMN].values, test_seqs, None, word_index
    else:
        return train_seqs, train_df[LABEL_COLUMN].values, test_seqs, test_df[LABEL_COLUMN].values, word_index


def get_text_only_dataset_with_graph_features(df, tweets_path, clef_submission=0):
    tweets_graph = TweetsGraph(tweets_path)

    result = []
    for _, row in df.iterrows():
        try:
            feature_nodes = tweets_graph.get_feature_nodes(row["tweet_id"])
        except:
            feature_nodes = []

        row_dict = {}
        row_dict["tweet_id"] = row["tweet_id"]
        row_dict["topic_id"] = row["topic_id"]
        if clef_submission == 0:
            row_dict["check_worthiness"] = row["check_worthiness"]
        row_dict["tweet_text"] = row["tweet_text"]

        row_dict["rel_tweet_id_1"] = 0
        row_dict["rel_text_1"] = ""
        row_dict["rel_tweet_id_2"] = 0
        row_dict["rel_text_2"] = ""
        row_dict["rel_tweet_id_3"] = 0
        row_dict["rel_text_3"] = ""
        if len(feature_nodes) > 0:
            row_dict["rel_tweet_id_1"] = feature_nodes[0]
            rltx1 = df.loc[df["tweet_id"] ==
                           feature_nodes[0]]["tweet_text"].values
            row_dict["rel_text_1"] = rltx1[0] if len(rltx1) else ''
        if len(feature_nodes) > 1:
            row_dict["rel_tweet_id_2"] = feature_nodes[1]
            rltx2 = df.loc[df["tweet_id"] ==
                           feature_nodes[1]]["tweet_text"].values
            row_dict["rel_text_2"] = rltx2[0] if len(rltx2) else ''
        if len(feature_nodes) > 2:
            row_dict["rel_tweet_id_3"] = feature_nodes[2]
            rltx3 = df.loc[df["tweet_id"] ==
                           feature_nodes[2]]["tweet_text"].values
            row_dict["rel_text_3"] = rltx3[0] if len(rltx3) else ''

        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def clean_dataset(df, clef_submission=0):
    result = []
    for _, row in df.iterrows():
        row_dict = {}
        row_dict["tweet_id"] = row["tweet_id"]
        row_dict["topic_id"] = row["topic_id"]
        if clef_submission == 0:
            row_dict["check_worthiness"] = row["check_worthiness"]
        row_dict["tweet_text"] = row["tweet_text"]

        row_dict["rel_tweet_id_1"] = row["rel_tweet_id_1"]
        row_dict["rel_text_1"] = row["rel_text_1"]
        row_dict["rel_tweet_id_2"] = row["rel_tweet_id_2"]
        row_dict["rel_text_2"] = row["rel_text_2"]
        row_dict["rel_tweet_id_3"] = row["rel_tweet_id_3"]
        row_dict["rel_text_3"] = row["rel_text_3"]

        if row["rel_tweet_id_1"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_1"]]) == 0:
                row_dict["rel_text_1"] = ""
        if row["rel_tweet_id_2"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_2"]]) == 0:
                row_dict["rel_text_2"] = ""
        if row["rel_tweet_id_3"] != 0:
            if len(df.loc[df["tweet_id"] == row["rel_tweet_id_3"]]) == 0:
                row_dict["rel_text_3"] = ""

        result.append(row_dict)
    result_df = pd.DataFrame(result)
    return result_df


def get_embedding_layer(pretrained_embedding_path, word_index):
    pretrained_np_embedding_path = pretrained_embedding_path.split(".")[
        0] + ".npy"

    if path.exists(pretrained_np_embedding_path):
        embedding_matrix = np.load(pretrained_np_embedding_path)
        print('in get_embedding_layer:', embedding_matrix.shape)
    else:
        print("Loading embeddings...")
        embeddings_index = {}
        embeddings_file = open(pretrained_embedding_path)
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

        np.save(pretrained_np_embedding_path, embedding_matrix)
        print("Loaded!")

    return embedding_matrix


def tfidf_transform(train_df, test_df, text_column):
    tiv = TfidfVectorizer(ngram_range=(1, 1))
    # tiv = HashingVectorizer(n_features=500)
    train_tiv_fit = tiv.fit(train_df[text_column])
    train_tiv_fit_transform = train_tiv_fit.transform(train_df[text_column])
    train_x = train_tiv_fit_transform.toarray()
    test_x = train_tiv_fit.transform(test_df[text_column]).toarray()
    return train_x, test_x


def get_tfidf(dataset, train_df, test_df, clef_submission=0):
    label_col = COL_NAMES[dataset]['labelc']
    text_col = COL_NAMES[dataset]['textc']
    train_x, test_x = tfidf_transform(train_df, test_df, text_col)
    train_y = train_df.pop(label_col).values
    if clef_submission == 1:
        test_y = None
    else:
        test_y = test_df.pop(label_col).values
    return train_x, train_y, test_x, test_y


def get_tfidf_gf(train_df, test_df, clef_submission=0):
    train_0_x, test_0_x = tfidf_transform(train_df, test_df, TEXT_COLUMN)
    rel_train_1_x, rel_test_1_x = tfidf_transform(
        train_df, test_df, "rel_text_1")
    rel_train_2_x, rel_test_2_x = tfidf_transform(
        train_df, test_df, "rel_text_2")
    rel_train_3_x, rel_test_3_x = tfidf_transform(
        train_df, test_df, "rel_text_3")

    train_x = np.concatenate(
        (train_0_x, rel_train_1_x, rel_train_2_x, rel_train_3_x), axis=1)
    test_x = np.concatenate(
        (test_0_x, rel_test_1_x, rel_test_2_x, rel_test_3_x), axis=1)

    train_y = train_df.pop(LABEL_COLUMN).values
    if clef_submission == 1:
        test_y = None
    else:
        test_y = test_df.pop(LABEL_COLUMN).values

    return train_x, train_y, test_x, test_y
