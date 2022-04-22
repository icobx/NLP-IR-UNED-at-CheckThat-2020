import os


# DATA_DIR_PATH = '/Users/icobx/Documents/skola/dp/code/data'
DATA_DIR_PATH = '/home/jovyan/sharedstorage/s12b3v/dp/data'
# TRAIN_PATH = "data/v2.0/training/"
TRAIN_PATH = os.path.join(DATA_DIR_PATH, 'political_debates', 'train')
# TEST_PATH = "test-input/test-input/"
TEST_PATH = os.path.join(DATA_DIR_PATH, 'political_debates', 'test')

# EXP_DIR_PATH = '/home/jovyan/sharedstorage/s12b3v/dp/dt/exp'
# LOG_DIR_PATH = '/home/jovyan/sharedstorage/s12b3v/dp/dt/log'
# BERT_MODEL_PATH = '/home/jovyan/sharedstorage/s12b3v/dp/dt/bert_models'
# SBERT_MODEL_PATH = '/home/jovyan/sharedstorage/s12b3v/dp/dt/sbert_models'
# SPACY_MODEL_PATH = '/home/jovyan/sharedstorage/s12b3v/dp/dt/spacy_models'

# POLIT_DATA_DIR_PATH = p.join(DATA_DIR_PATH, 'political_debates')
# COVID_DATA_DIR_PATH = p.join(DATA_DIR_PATH, 'covid_tweets')

SUMMARY_DATA_PATH = "nlpir01/results/summary/"
TRAINING_RESULTS_PATH = "results/training_debates/"
SUBMISSION_RESULTS_PATH = "results/submission_debates/"

RESOURCES_PATH = "nlpir01/resources/"
EMBEDDINGS_FILE = "glove.6B.100d.txt"

COL_NAMES = ["line_number", "speaker", "text", "label"]
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


BATCH_SIZE = 64
EPOCHS = 20

NUM_WORDS = 15000
SEQ_LEN = 100
EMBEDDING_SIZE = 200
