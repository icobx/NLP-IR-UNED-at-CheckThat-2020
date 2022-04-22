from utils.global_parameters import COL_NAMES, LABEL_COLUMN, EMBEDDINGS_FILE
from utils.global_parameters import TEST_PATH, SUBMISSION_RESULTS_PATH, RESOURCES_PATH
from utils.global_parameters import TRAIN_PATH, TRAINING_RESULTS_PATH, SUMMARY_DATA_PATH
from utils.models import get_bilstm_emb_model, get_ffnn_model
from utils.models import get_ffnn_emb_model, get_cnn_emb_model, get_lstm_emb_model
from utils.model_utils import evaluate_model_rid, predict_model_text, evaluate_model_text
from utils.model_utils import fit_model, evaluate_model, predict_model, predict_model_rid
from utils.model_utils import get_sequences_from_dataset, get_rid_counts_from_dataset
from utils.model_utils import run_model, run_model_rid, run_model_text, get_embedding_layer
from utils.model_utils import check_and_evaluate, get_train_text_features
from utils.text_features import get_feature_matrix, get_pos_df
from utils.path_utils import prepare_folders_and_files
from utils.path_utils import get_dataframe_from_file, get_dataframe_from_file_list, get_full_path
from utils.path_utils import get_file_list, append_path_to_file_list, get_file_with_path_list
from datetime import datetime
from sklearn.metrics import classification_report
from os import listdir
from os.path import join, dirname
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


sys.path.append(os.path.abspath("."))


MODEL1_PREFIX_LABEL = "Model 1: Naive"
MODEL2_PREFIX_LABEL = "Model 2: Dense"
MODEL3_PREFIX_LABEL = "Model 3: CNN"
MODEL4_PREFIX_LABEL = "Model 4: LSTM"
MODEL5_PREFIX_LABEL = "Model 5: Bi-LSTM"
MODEL6_PREFIX_LABEL = "Model 6: Dense TF-IDF"
MODEL7_PREFIX_LABEL = "Model 7: Dense RID"
MODEL1_SHORT_LABEL = "nlpir01run1"
MODEL2_SHORT_LABEL = "nlpir01run2"
MODEL3_SHORT_LABEL = "nlpir01run3"
MODEL4_SHORT_LABEL = "nlpir01run4"
MODEL5_SHORT_LABEL = "nlpir01run5"
MODEL6_SHORT_LABEL = "nlpir01run6"
MODEL7_SHORT_LABEL = "nlpir01run7"


def build_subplot(model_label, history_dict):
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.title(model_label)
    plt.plot(epochs, loss, "r-", label="Training loss")
    plt.plot(epochs, acc, "y-", label="Training acc")
    plt.plot(epochs, val_loss, "g-", label="Validation loss")
    plt.plot(epochs, val_acc, "b-", label="Validation acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.legend()


def get_initial_data():
    alltrain_debates = get_file_list(TRAIN_PATH)
    alltrain_debates.sort()
    train_partition = int(.8 * len(alltrain_debates))
    train_debates = alltrain_debates  # [:train_partition]
    train_debates = append_path_to_file_list(train_debates, TRAIN_PATH)
    test_debates = get_file_list(TEST_PATH)
    # alltrain_debates[train_partition:]
    # test_path = TRAIN_PATH
    test_path = TEST_PATH
    training_results_path = get_full_path(TRAINING_RESULTS_PATH)

    test_submission_debates = get_file_list(TEST_PATH)
    test_submission_debates.sort()
    submission_results_path = get_full_path(SUBMISSION_RESULTS_PATH)

    return train_debates, test_debates, test_path, training_results_path, test_submission_debates, submission_results_path


def get_sequenced_datasets(train_debates, test_debates, oversampling=0):
    train_df = get_dataframe_from_file_list(train_debates, COL_NAMES)

    if oversampling == 1:
        # --- random oversampling class 1 ---
        count_class_0, count_class_1 = train_df.label.value_counts()
        print("Class 0 count: ", count_class_0, ", class 1 count:", count_class_1)
        class_0_df = train_df[train_df[LABEL_COLUMN] == 0]
        class_1_df = train_df[train_df[LABEL_COLUMN] == 1]
        class_1_over_df = class_1_df.sample(count_class_0, replace=True)
        train_df = pd.concat([class_0_df, class_1_over_df], axis=0)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        count_class_0, count_class_1 = train_df.label.value_counts()
        print("After resampling, class 0 count: ", count_class_0, ", class 1 count:", count_class_1)

    # test_debates1 = append_path_to_file_list(test_debates, TRAIN_PATH)
    test_debates1 = append_path_to_file_list(test_debates, TEST_PATH)
    test1_df = get_dataframe_from_file_list(test_debates1, COL_NAMES)  # test1_df only to check accuracy
    x_train, y_train, _, _, word_index = get_sequences_from_dataset(train_df, train_df)
    input_length = x_train[:].shape[1]
    return input_length, x_train, y_train, train_df, test1_df, word_index


def get_rid_datasets(train_df):
    x_train_rid, _ = get_rid_counts_from_dataset(train_df, train_df)
    input_length_rid = x_train_rid[:].shape[1]
    return input_length_rid, x_train_rid


def get_text_datasets(train_df):
    x_train_text, train_tiv_fit, train_pos_tiv_fit = get_train_text_features(train_df)
    input_length_text = x_train_text[:].shape[1]
    return input_length_text, x_train_text, train_tiv_fit, train_pos_tiv_fit


def run_task5(models_to_run, embeddings=0, oversampling=0, verbose=0):

    prepare_folders_and_files()

    if embeddings == 1:
        label = "glove"
    else:
        label = "autoemb"

    if oversampling == 1:
        if label == "":
            label = "oversampling"
        else:
            label = label + "_oversampling"

    model1_label = MODEL1_PREFIX_LABEL + " + " + label
    model2_label = MODEL2_PREFIX_LABEL + " + " + label
    model3_label = MODEL3_PREFIX_LABEL + " + " + label
    model4_label = MODEL4_PREFIX_LABEL + " + " + label
    model5_label = MODEL5_PREFIX_LABEL + " + " + label
    model6_label = MODEL6_PREFIX_LABEL + " + " + label
    model7_label = MODEL7_PREFIX_LABEL + " + " + label

    train_debates, test_debates, test_path, training_results_path, test_submission_debates, submission_results_path = get_initial_data()
    input_length, x_train, y_train, train_df, test1_df, word_index = get_sequenced_datasets(train_debates, test_debates,
                                                                                            oversampling=oversampling)
    # print(test1_df)
    input_length_rid, x_train_rid = get_rid_datasets(train_df)
    input_length_text, x_train_text, train_tiv_fit, train_pos_tiv_fit = get_text_datasets(train_df)

    embedding_matrix = None
    if embeddings == 1:
        embedding_matrix = get_embedding_layer(
            os.path.join(RESOURCES_PATH, EMBEDDINGS_FILE), word_index)

    if 1 in models_to_run:
        model1 = get_ffnn_emb_model(embedding_matrix, word_index, input_length, 0,
                                    activation="sigmoid", optimizer="adam", verbose=verbose)
        model1, history_dict1 = fit_model(model1, x_train, y_train, batch_size=8, epochs=50, verbose=verbose)
        loss1, accuracy1, class_predictions1, y_test1 = evaluate_model(model1, train_df, test1_df, verbose=0)
        predict_model(model1, train_df, test_debates, test_path, training_results_path, model1_label)
        result1 = check_and_evaluate(test_debates, test_path, training_results_path, model1_label)
        print(model1_label + "-nlpir01", "MAP AVGP:", result1)
        predict_model(model1, train_df, test_submission_debates, TEST_PATH, submission_results_path, model1_label)

    if 2 in models_to_run:
        model2 = get_ffnn_emb_model(embedding_matrix, word_index, input_length, 1000,
                                    activation="sigmoid", optimizer="adam", verbose=verbose)
        model2, history_dict2 = fit_model(model2, x_train, y_train, verbose=verbose)
        loss2, accuracy2, class_predictions2, y_test1 = evaluate_model(model2, train_df, test1_df, verbose=0)
        predict_model(model2, train_df, test_debates, test_path, training_results_path, model2_label)
        result2 = check_and_evaluate(test_debates, test_path, training_results_path, model2_label)
        print(model2_label + "-nlpir01", "MAP AVGP:", result2)
        predict_model(model2, train_df, test_submission_debates, TEST_PATH, submission_results_path, model2_label)

    if 3 in models_to_run:
        model3 = get_cnn_emb_model(embedding_matrix, word_index, input_length, 256,
                                   activation="sigmoid", optimizer="adam", verbose=verbose)
        model3, history_dict3 = fit_model(model3, x_train, y_train, verbose=verbose)
        loss3, accuracy3, class_predictions3, y_test1 = evaluate_model(model3, train_df, test1_df, verbose=0)
        predict_model(model3, train_df, test_debates, test_path, training_results_path, model3_label)
        result3 = check_and_evaluate(test_debates, test_path, training_results_path, model3_label)
        print(model3_label + "-nlpir01", "MAP AVGP:", result3)
        predict_model(model3, train_df, test_submission_debates, TEST_PATH, submission_results_path, model3_label)

    if 4 in models_to_run:
        model4 = get_lstm_emb_model(embedding_matrix, word_index, input_length,
                                    dropout=0.2, recurrent_dropout=0, optimizer="adam", verbose=verbose)
        model4, history_dict4 = fit_model(model4, x_train, y_train, verbose=verbose)
        loss4, accuracy4, class_predictions4, y_test1 = evaluate_model(model4, train_df, test1_df, verbose=0)
        predict_model(model4, train_df, test_debates, test_path, training_results_path, model4_label)
        result4 = check_and_evaluate(test_debates, test_path, training_results_path, model4_label)
        print(model4_label + "-nlpir01", "MAP AVGP:", result4)
        predict_model(model4, train_df, test_submission_debates, TEST_PATH, submission_results_path, model4_label)

    if 5 in models_to_run:
        model5 = get_bilstm_emb_model(embedding_matrix, word_index, input_length, hidden_layer_size=256,
                                      dropout=0.2, recurrent_dropout=0, optimizer="adam", verbose=verbose)
        model5, history_dict5 = fit_model(model5, x_train, y_train, batch_size=16, epochs=10, verbose=verbose)
        loss5, accuracy5, class_predictions5, y_test1 = evaluate_model(model5, train_df, test1_df, verbose=0)
        predict_model(model5, train_df, test_debates, test_path, training_results_path, model5_label)
        result5 = check_and_evaluate(test_debates, test_path, training_results_path, model5_label)
        print(model5_label + "-nlpir01", "MAP AVGP:", result5)
        predict_model(model5, train_df, test_submission_debates, TEST_PATH, submission_results_path, model5_label)

    if 6 in models_to_run:
        model6 = get_ffnn_emb_model(embedding_matrix, word_index, input_length_text, 0,
                                    activation="sigmoid", optimizer="adam", verbose=verbose)
        model6, history_dict6 = fit_model(model6, x_train_text, y_train, verbose=verbose)
        loss6, accuracy6, class_predictions6, y_test1 = evaluate_model_text(model6, train_df, test1_df,
                                                                            train_tiv_fit, train_pos_tiv_fit, verbose=0)
        predict_model_text(model6, train_df, test_debates, test_path, training_results_path, model6_label,
                           train_tiv_fit, train_pos_tiv_fit)
        result6 = check_and_evaluate(test_debates, test_path, training_results_path, model6_label)
        print(model6_label + "-nlpir01", "MAP AVGP:", result6)
        predict_model_text(model6, train_df, test_submission_debates, TEST_PATH, submission_results_path, model6_label,
                           train_tiv_fit, train_pos_tiv_fit)

    if 7 in models_to_run:
        model7 = get_ffnn_model(input_length_rid, 102, activation="elu", optimizer="adam", verbose=verbose)
        model7, history_dict7 = fit_model(model7, x_train_rid, y_train, verbose=verbose)
        loss7, accuracy7, class_predictions7, y_test1 = evaluate_model_rid(model7, train_df, test1_df, verbose=0)
        predict_model_rid(model7, train_df, test_debates, test_path, training_results_path, model7_label)
        result7 = check_and_evaluate(test_debates, test_path, training_results_path, model7_label)
        print(model7_label + "-nlpir01", "MAP AVGP:", result7)
        predict_model_rid(model7, train_df, test_submission_debates, TEST_PATH, submission_results_path, model7_label)

    with open(join(SUMMARY_DATA_PATH, "results_2020_" + label + ".md"), "w") as results_file:

        results_file.write("# 2020 models result\n")
        results_file.write(f"total time: {datetime.now() - starting_time}\n")

        if 1 in models_to_run:
            results_file.write("## " + model1_label + "\n")
            results_file.write(f"map avgp: {result1}\n")
            results_file.write(f"test loss: {loss1} test accuracy: {accuracy1}\n")
            results_file.write(classification_report(y_test1, class_predictions1, zero_division=0) + "\n")

        if 2 in models_to_run:
            results_file.write("## " + model2_label + "\n")
            results_file.write(f"map avgp: {result2}\n")
            results_file.write(f"test loss: {loss2} test accuracy: {accuracy2}\n")
            results_file.write(classification_report(y_test1, class_predictions2, zero_division=0) + "\n")

        if 3 in models_to_run:
            results_file.write("## " + model3_label + "\n")
            results_file.write(f"map avgp: {result3}\n")
            results_file.write(f"test loss: {loss3} test accuracy: {accuracy3}\n")
            results_file.write(classification_report(y_test1, class_predictions3, zero_division=0) + "\n")

        if 4 in models_to_run:
            results_file.write("## " + model4_label + "\n")
            results_file.write(f"map avgp: {result4}\n")
            results_file.write(f"test loss: {loss4} test accuracy: {accuracy4}\n")
            results_file.write(classification_report(y_test1, class_predictions4, zero_division=0) + "\n")

        if 5 in models_to_run:
            results_file.write("## " + model5_label + "\n")
            results_file.write(f"map avgp: {result5}\n")
            results_file.write(f"test loss: {loss5} test accuracy: {accuracy5}\n")
            results_file.write(classification_report(y_test1, class_predictions5, zero_division=0) + "\n")

        if 6 in models_to_run:
            results_file.write("## " + model6_label + "\n")
            results_file.write(f"map avgp: {result6}\n")
            results_file.write(f"test loss: {loss6} test accuracy: {accuracy6}\n")
            results_file.write(classification_report(y_test1, class_predictions6, zero_division=0) + "\n")

        if 7 in models_to_run:
            results_file.write("## " + model7_label + "\n")
            results_file.write(f"map avgp: {result7}\n")
            results_file.write(f"test loss: {loss7} test accuracy: {accuracy7}\n")
            results_file.write(classification_report(y_test1, class_predictions7, zero_division=0) + "\n")

    plt.figure(figsize=(12, 9))

    if 1 in models_to_run:
        plt.subplot(331)
        build_subplot(model1_label, history_dict1)

    if 2 in models_to_run:
        plt.subplot(332)
        build_subplot(model2_label, history_dict2)

    if 3 in models_to_run:
        plt.subplot(333)
        build_subplot(model3_label, history_dict3)

    if 4 in models_to_run:
        plt.subplot(334)
        build_subplot(model4_label, history_dict4)

    if 5 in models_to_run:
        plt.subplot(335)
        build_subplot(model5_label, history_dict5)

    if 6 in models_to_run:
        plt.subplot(336)
        build_subplot(model6_label, history_dict6)

    if 7 in models_to_run:
        plt.subplot(337)
        build_subplot(model6_label, history_dict7)

    plt.tight_layout()
    plt.savefig(join(SUMMARY_DATA_PATH, "results_2020_" + label + ".png"))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--check_all", "-ca", type=int, default=0, choices=[0, 1],
    #     help="0 = run a specific configuration, 1 = run all combinations of embeddings and oversampling")
    # parser.add_argument("--models", "-m", nargs="+", type=int, default=[5, 7],
    #     help="List of models to run. All models = [1, 2, 3, 4, 5, 6, 7]")
    # parser.add_argument("--embeddings", "-e", type=int, default=1, choices=[0, 1],
    #     help="0 = auto-generated embeddings, 1 = Glove embeddings.")
    # parser.add_argument("--oversampling", "-o", type=int, default=0, choices=[0, 1],
    #     help="0 = do not use oversampling, 1 = use oversampling.")
    # args = parser.parse_args()

    starting_time = datetime.now()

    # if args.check_all == 0:
    #     run_task5(args.models, embeddings=args.embeddings, oversampling=args.oversampling)
    # else:
    #     run_task5(args.models, embeddings=1, oversampling=1)
    #     run_task5(args.models, embeddings=1, oversampling=0)
    #     run_task5(args.models, embeddings=0, oversampling=1)
    #     run_task5(args.models, embeddings=0, oversampling=0)
    models_ids = [i for i in range(1, 8)]
    models = models_ids[4:5]
    print(models)
    run_task5(models, embeddings=1, oversampling=0, verbose=1)
    ending_time = datetime.now()
    print("Total time:", ending_time - starting_time)
    print("Done!!!")
