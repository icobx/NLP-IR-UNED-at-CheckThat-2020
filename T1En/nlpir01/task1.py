import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from sklearn.metrics import classification_report
from datetime import datetime
from os.path import join, dirname

sys.path.append(os.path.abspath("."))

from scorer.main import evaluate
from format_checker.main import check_format

from utils.models import get_ffnn_emb_model, get_cnn_emb_model, get_lstm_emb_model
from utils.models import get_bilstm_emb_model, get_ffnn_model
from utils.model_utils import get_embedding_layer, get_sequences_from_dataset, clean_dataset
from utils.model_utils import run_model_binary, export_results, get_tfidf, get_tfidf_gf
from utils.model_utils import get_text_only_dataset_with_graph_features, get_sequences_from_dataset_gf
from utils.model_utils import fit_model_binary, evaluate_model_binary, predict_model_binary
from utils.global_parameters import RESOURCES_PATH, RESULTS_PATH, SEQ_LEN, EMBEDDINGS_FILE
from utils.global_parameters import TRAINING_PATH, DEV_PATH, TEST_PATH, RESULTS_FILE_PREFIX
from utils.global_parameters import TRAINING_TWEETS_PATH, DEV_TWEETS_PATH, TEST_TWEETS_PATH
from utils.path_utils import prepare_folders_and_files



MODEL1_PREFIX_LABEL = "Model 1: Dense"
MODEL2_PREFIX_LABEL = "Model 2: CNN"
MODEL3_PREFIX_LABEL = "Model 3: LSTM"
MODEL4_PREFIX_LABEL = "Model 4: Bi-LSTM"
MODEL5_PREFIX_LABEL = "Model 5: Dense TF-IDF"
MODEL6_PREFIX_LABEL = "Model 6: "
MODEL1_SHORT_LABEL = "nlpir01-run1"
MODEL2_SHORT_LABEL = "nlpir01-run2"
MODEL3_SHORT_LABEL = "nlpir01-run3"
MODEL4_SHORT_LABEL = "nlpir01-run4"
MODEL5_SHORT_LABEL = "nlpir01-run5"
MODEL6_SHORT_LABEL = "nlpir01-run6"


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


def get_data(graph_features=0, use_embeddings=1, clef_submission=0):
    train_df = pd.read_csv(TRAINING_PATH, sep='\t')
    test_df = pd.read_csv(DEV_PATH, sep='\t')
    train_sub_df = train_df.copy() # hard copy because label is extracted later from test_df
    test_sub_df = pd.read_csv(TEST_PATH, sep='\t')

    word_index = None
    x_test_sub = None

    if graph_features == 0:
        if use_embeddings == 1:
            x_train, y_train, x_test, y_test, word_index = get_sequences_from_dataset(
                train_df, test_df, clef_submission=0)
            if clef_submission == 1:
                _, _, x_test_sub, _, _ = get_sequences_from_dataset(
                    train_sub_df, test_sub_df, clef_submission=1)
        else:
            x_train, y_train, x_test, y_test = get_tfidf(train_df, test_df, clef_submission=0)
            if clef_submission == 1:
                _, _, x_test_sub, _ = get_tfidf(train_sub_df, test_sub_df, clef_submission=1)

        input_length = x_train[:].shape[1]

    else:
        train_df = get_text_only_dataset_with_graph_features(train_df, TRAINING_TWEETS_PATH)
        test_df = get_text_only_dataset_with_graph_features(test_df, DEV_TWEETS_PATH)
        train_sub_df = train_df.copy() # hard copy because label is extracted later from test_df
        test_sub_df = get_text_only_dataset_with_graph_features(test_sub_df, TEST_TWEETS_PATH, clef_submission=1)

        train_df = clean_dataset(train_df)
        test_df = clean_dataset(test_df)
        test_sub_df = clean_dataset(test_sub_df, clef_submission=1)

        if use_embeddings == 1:
            x_train, y_train, x_test, y_test, word_index = get_sequences_from_dataset_gf(
                train_df, test_df, clef_submission=0)
            if clef_submission == 1:
                _, _, x_test_sub, _, _ = get_sequences_from_dataset_gf(
                    train_sub_df, test_sub_df, clef_submission=1)
        else:
            x_train, y_train, x_test, y_test = get_tfidf_gf(
                train_df, test_df, clef_submission=0)
            if clef_submission == 1:
                _, _, x_test_sub, _ = get_tfidf_gf(
                    train_sub_df, test_sub_df, clef_submission=1)

        input_length = x_train[:].shape[1]


    if use_embeddings == 1:
        print("Embeddings input:")
    else:
        print("TF-IDF input:")
    print("\tShape of x_train tensor:", x_train.shape)
    print("\tShape of class tensor:", y_train.shape)
    print("\tInput lenght:", input_length)

    return input_length, x_train, y_train, x_test, y_test, test_df, word_index, x_test_sub, test_sub_df


def run_task1(models_to_run, graph_features=1, embeddings=0, verbose=0):
    
    prepare_folders_and_files()

    if embeddings == 1:
        label_embbedings = "glove"
    else:
        label_embbedings = "autoemb"

    if graph_features == 1:
        label_graph = "graph"
    else:
        label_graph = "nograph"

    model1_label = MODEL1_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model2_label = MODEL2_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model3_label = MODEL3_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model4_label = MODEL4_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model5_label = MODEL5_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    model6_label = MODEL6_PREFIX_LABEL + " + " + label_embbedings + " + " + label_graph
    TEST_LABEL = "_only_for_testing"

    input_length, x_train, y_train, x_test, y_test, test_df, word_index, x_test_sub, test_sub_df = get_data(graph_features, use_embeddings=1, clef_submission=1)
    input_length2, x_train2, y_train2, x_test2, y_test2, test_df, _, x_test_sub2, test_sub_df = get_data(graph_features, use_embeddings=0, clef_submission=1)

    embedding_matrix = None
    if embeddings == 1:
        embedding_matrix = get_embedding_layer(
            os.path.join(RESOURCES_PATH, EMBEDDINGS_FILE), word_index)

    if 1 in models_to_run:
        print("Running", model1_label)
        model1 = get_ffnn_emb_model(embedding_matrix, word_index, input_length, 1000,
                            activation="hard_sigmoid", optimizer="adam", verbose=verbose)

        model1, history_dict1 = fit_model_binary(model1, x_train, y_train, epochs=100, batch_size=8, verbose=verbose)
        loss1, accuracy1 = evaluate_model_binary(model1, x_test, y_test, verbose=verbose)
        class_predictions1, raw_predictions1 = predict_model_binary(model1, x_test)
        export_results(test_df, raw_predictions1, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL1_SHORT_LABEL + TEST_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL1_SHORT_LABEL + TEST_LABEL)
        if y_test is not None:
            if check_format(MODEL_RESULT_PATH):
                _, _, avg_precision1, _, _ = evaluate(DEV_PATH, MODEL_RESULT_PATH)

        _, raw_predictions_sub1 = predict_model_binary(model1, x_test_sub)
        export_results(test_sub_df, raw_predictions_sub1, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL1_SHORT_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL1_SHORT_LABEL)
        if not check_format(MODEL_RESULT_PATH):
            print("Format error:", MODEL_RESULT_PATH)


    if 2 in models_to_run:
        print("Running", model2_label)
        model2 = get_cnn_emb_model(embedding_matrix, word_index, input_length, 32,
                            activation="sigmoid", optimizer="adam", verbose=verbose)

        model2, history_dict2 = fit_model_binary(model2, x_train, y_train, epochs=10, batch_size=16, verbose=verbose)
        loss2, accuracy2 = evaluate_model_binary(model2, x_test, y_test, verbose=verbose)
        class_predictions2, raw_predictions2 = predict_model_binary(model2, x_test)
        export_results(test_df, raw_predictions2, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL2_SHORT_LABEL + TEST_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL2_SHORT_LABEL + TEST_LABEL)
        if y_test is not None:
            if check_format(MODEL_RESULT_PATH):
                _, _, avg_precision2, _, _ = evaluate(DEV_PATH, MODEL_RESULT_PATH)

        _, raw_predictions_sub2 = predict_model_binary(model2, x_test_sub)
        export_results(test_sub_df, raw_predictions_sub2, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL2_SHORT_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL2_SHORT_LABEL)
        if not check_format(MODEL_RESULT_PATH):
            print("Format error:", MODEL_RESULT_PATH)


    if 3 in models_to_run:
        print("Running", model3_label)
        model3 = get_lstm_emb_model(embedding_matrix, word_index, input_length,
                    dropout=0.1, recurrent_dropout=0, optimizer="adam", verbose=verbose)

        model3, history_dict3 = fit_model_binary(model3, x_train, y_train, epochs=5, batch_size=8, verbose=verbose)
        loss3, accuracy3 = evaluate_model_binary(model3, x_test, y_test, verbose=verbose)
        class_predictions3, raw_predictions3 = predict_model_binary(model3, x_test)
        export_results(test_df, raw_predictions3, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL3_SHORT_LABEL + TEST_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL3_SHORT_LABEL + TEST_LABEL)
        if y_test is not None:
            if check_format(MODEL_RESULT_PATH):
                _, _, avg_precision3, _, _ = evaluate(DEV_PATH, MODEL_RESULT_PATH)

        _, raw_predictions_sub3 = predict_model_binary(model3, x_test_sub)
        export_results(test_sub_df, raw_predictions_sub3, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL3_SHORT_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL3_SHORT_LABEL)
        if not check_format(MODEL_RESULT_PATH):
            print("Format error:", MODEL_RESULT_PATH)


    if 4 in models_to_run:
        print("Running", model4_label)
        model4 = get_bilstm_emb_model(embedding_matrix, word_index, input_length, hidden_layer_size=10,
                    dropout=0.1, recurrent_dropout=0, activation="tanh", optimizer="nadam", verbose=verbose)

        model4, history_dict4 = fit_model_binary(model4, x_train, y_train, epochs=10, batch_size=64, verbose=verbose)
        loss4, accuracy4 = evaluate_model_binary(model4, x_test, y_test, verbose=verbose)
        class_predictions4, raw_predictions4 = predict_model_binary(model4, x_test)
        export_results(test_df, raw_predictions4, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL4_SHORT_LABEL + TEST_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL4_SHORT_LABEL + TEST_LABEL)
        if y_test is not None:
            if check_format(MODEL_RESULT_PATH):
                _, _, avg_precision4, _, _ = evaluate(DEV_PATH, MODEL_RESULT_PATH)

        _, raw_predictions_sub4 = predict_model_binary(model4, x_test_sub)
        export_results(test_sub_df, raw_predictions_sub4, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL4_SHORT_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL4_SHORT_LABEL)
        if not check_format(MODEL_RESULT_PATH):
            print("Format error:", MODEL_RESULT_PATH)


    if 5 in models_to_run:
        print("Running", model5_label)
        model5 = get_ffnn_model(input_length2, 1000,
                    activation="sigmoid", optimizer="adam", verbose=verbose)

        model5, history_dict5 = fit_model_binary(model5, x_train2, y_train2, epochs=50, batch_size=8, verbose=verbose)
        loss5, accuracy5 = evaluate_model_binary(model5, x_test2, y_test2, verbose=verbose)
        class_predictions5, raw_predictions5 = predict_model_binary(model5, x_test2)
        export_results(test_df, raw_predictions5, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL5_SHORT_LABEL + TEST_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL5_SHORT_LABEL + TEST_LABEL)
        if y_test2 is not None:
            if check_format(MODEL_RESULT_PATH):
                _, _, avg_precision5, _, _ = evaluate(DEV_PATH, MODEL_RESULT_PATH)

        _, raw_predictions_sub5 = predict_model_binary(model5, x_test_sub2)
        export_results(test_sub_df, raw_predictions_sub5, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL5_SHORT_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL5_SHORT_LABEL)
        if not check_format(MODEL_RESULT_PATH):
            print("Format error:", MODEL_RESULT_PATH)


    if 6 in models_to_run:
        print("Running", model6_label)
        model6 = None

        model6, history_dict6 = fit_model_binary(model6, x_train, y_train, epochs=10, batch_size=64, verbose=verbose)
        loss6, accuracy6 = evaluate_model_binary(model6, x_test, y_test, verbose=verbose)
        class_predictions6, raw_predictions6 = predict_model_binary(model6, x_test)
        export_results(test_df, raw_predictions6, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL6_SHORT_LABEL + TEST_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL6_SHORT_LABEL + TEST_LABEL)
        if y_test is not None:
            if check_format(MODEL_RESULT_PATH):
                _, _, avg_precision6, _, _ = evaluate(DEV_PATH, MODEL_RESULT_PATH)

        _, raw_predictions_sub6 = predict_model_binary(model6, x_test_sub)
        export_results(test_sub_df, raw_predictions_sub6, RESULTS_PATH, RESULTS_FILE_PREFIX, MODEL6_SHORT_LABEL)
        MODEL_RESULT_PATH = join(RESULTS_PATH, RESULTS_FILE_PREFIX + MODEL6_SHORT_LABEL)
        if not check_format(MODEL_RESULT_PATH):
            print("Format error:", MODEL_RESULT_PATH)



    if y_test is not None:
        with open(join(RESULTS_PATH, RESULTS_FILE_PREFIX + "results_" + label_embbedings + "_" + label_graph + ".txt"), "w") as results_file:
            if 1 in models_to_run:
                results_file.write(model1_label + "\n")
                results_file.write(f"test loss: {loss1} test accuracy: {accuracy1}\n")
                results_file.write(f"avg precision: {avg_precision1}\n")
                results_file.write(classification_report(y_test, class_predictions1) + "\n")

            if 2 in models_to_run:
                results_file.write(model2_label + "\n")
                results_file.write(f"test loss: {loss2} test accuracy: {accuracy2}\n")
                results_file.write(f"avg precision: {avg_precision2}\n")
                results_file.write(classification_report(y_test, class_predictions2) + "\n")

            if 3 in models_to_run:
                results_file.write(model3_label + "\n")
                results_file.write(f"test loss: {loss3} test accuracy: {accuracy3}\n")
                results_file.write(f"avg precision: {avg_precision3}\n")
                results_file.write(classification_report(y_test, class_predictions3) + "\n")

            if 4 in models_to_run:
                results_file.write(model4_label + "\n")
                results_file.write(f"test loss: {loss4} test accuracy: {accuracy4}\n")
                results_file.write(f"avg precision: {avg_precision4}\n")
                results_file.write(classification_report(y_test, class_predictions4) + "\n")

            if 5 in models_to_run:
                results_file.write(model5_label + "\n")
                results_file.write(f"test loss: {loss5} test accuracy: {accuracy5}\n")
                results_file.write(f"avg precision: {avg_precision5}\n")
                results_file.write(classification_report(y_test, class_predictions5) + "\n")

            if 6 in models_to_run:
                results_file.write(model6_label + "\n")
                results_file.write(f"test loss: {loss6} test accuracy: {accuracy6}\n")
                results_file.write(f"avg precision: {avg_precision6}\n")
                results_file.write(classification_report(y_test, class_predictions6) + "\n")

    if y_test is not None or y_test2 is not None:

        if 1 in models_to_run:
            print("\n" + model1_label)
            print(f"test loss: {loss1} test accuracy: {accuracy1}")
            print(f"avg precision: {avg_precision1}")
            print(classification_report(y_test, class_predictions1))

        if 2 in models_to_run:
            print("\n" + model2_label)
            print(f"test loss: {loss2} test accuracy: {accuracy2}")
            print(f"avg precision: {avg_precision2}")
            print(classification_report(y_test, class_predictions2))

        if 3 in models_to_run:
            print("\n" + model3_label)
            print(f"test loss: {loss3} test accuracy: {accuracy3}")
            print(f"avg precision: {avg_precision3}")
            print(classification_report(y_test, class_predictions3))

        if 4 in models_to_run:
            print("\n" + model4_label)
            print(f"test loss: {loss4} test accuracy: {accuracy4}")
            print(f"avg precision: {avg_precision4}")
            print(classification_report(y_test, class_predictions4))

        if 5 in models_to_run:
            print("\n" + model5_label)
            print(f"test loss: {loss5} test accuracy: {accuracy5}")
            print(f"avg precision: {avg_precision5}")
            print(classification_report(y_test, class_predictions5))

        if 6 in models_to_run:
            print("\n" + model6_label)
            print(f"test loss: {loss6} test accuracy: {accuracy6}")
            print(f"avg precision: {avg_precision6}")
            print(classification_report(y_test, class_predictions6))

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

        plt.tight_layout()
        plt.savefig(join(RESULTS_PATH, RESULTS_FILE_PREFIX + "results_" + label_embbedings + "_" + label_graph + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_all", "-ca", type=int, default=0, choices=[0, 1],
        help='0 = run a specific configuration, 1 = run all combinations of embeddings and graph')
    parser.add_argument("--models", "-m", type=list, default=[1, 2, 4],
        help='List of models to run. All models = [1, 2, 3, 4, 5, 6]')
    parser.add_argument("--embeddings", "-e", type=int, default=1, choices=[0, 1],
        help='0 = auto-generated embeddings, 1 = Glove embeddings.')
    parser.add_argument("--graph", "-g", type=int, default=1, choices=[0, 1],
        help='0 = do not use graph, 1 = use graph.')
    args = parser.parse_args()  

    starting_time = datetime.now()

    print(args)
    if args.check_all == 0:
        run_task1(args.models, embeddings=args.embeddings, graph_features=args.graph)
    else:
        run_task1(args.models, embeddings=1, graph_features=1)
        run_task1(args.models, embeddings=1, graph_features=0)
        run_task1(args.models, embeddings=0, graph_features=1)
        run_task1(args.models, embeddings=0, graph_features=0)

    ending_time = datetime.now()
    print("Total time:", ending_time - starting_time)
    print("Done!!!")

