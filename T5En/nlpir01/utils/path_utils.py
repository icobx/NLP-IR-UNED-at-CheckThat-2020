import pandas as pd
import os

from os.path import join, dirname
from os import listdir

from utils.global_parameters import TRAINING_RESULTS_PATH, SUBMISSION_RESULTS_PATH, SUMMARY_DATA_PATH


def get_dataframe_from_file_list(file_list, col_names):
    df_list = []
    for file_name in file_list:
        df = pd.read_csv(file_name, index_col=None, header=None, names=col_names, sep="\t")
        df_list.append(df)
    return pd.concat(df_list)


def get_dataframe_from_file(file_name, col_names):
    return pd.read_csv(file_name, index_col=None, header=None, names=col_names, sep="\t")


def get_full_path(path):
    root_dir = dirname(dirname(__file__))
    return join(root_dir, path)


def get_file_with_path_list(path):
    data_dir = get_full_path(path)
    all_files = [join(path, file_name) for file_name in listdir(data_dir)]
    all_files.sort()
    return all_files


def get_file_list(path):
    # all_files = [file_name for file_name in listdir(path)]
    all_files = [file_name for file_name in listdir(path) if '_combined.tsv' not in file_name]
    all_files.sort()
    return all_files


def append_path_to_file_list(file_list, path):
    all_files = [join(path, file_name) for file_name in file_list]
    all_files.sort()
    return all_files


def prepare_folders_and_files():
    training_results_path = get_full_path(TRAINING_RESULTS_PATH)
    if not os.path.exists(training_results_path):
        os.makedirs(training_results_path)

    submission_results_path = get_full_path(SUBMISSION_RESULTS_PATH)
    if not os.path.exists(submission_results_path):
        os.makedirs(submission_results_path)

    if not os.path.exists(SUMMARY_DATA_PATH):
        os.makedirs(SUMMARY_DATA_PATH)
