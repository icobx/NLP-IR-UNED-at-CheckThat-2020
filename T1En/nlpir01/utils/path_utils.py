import os
from os.path import join, dirname
from os import listdir

from utils.global_parameters import RESULTS_PATH, EMBEDDINGS_FILE_CT, EMBEDDINGS_FILE_PD, RESOURCES_PATH


def get_file_with_path_list(path):
    all_files = [join(path, file_name) for file_name in listdir(path)]
    all_files.sort()
    return all_files


def prepare_folders_and_files(dataset='covid_tweets'):
    emb_file = EMBEDDINGS_FILE_CT if dataset == 'covid_tweets' else EMBEDDINGS_FILE_PD

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    embeddings_binary = os.path.join(
        RESOURCES_PATH, emb_file).split(".")[0] + ".npy"
    if os.path.exists(embeddings_binary):
        os.remove(embeddings_binary)
