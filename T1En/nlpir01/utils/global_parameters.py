from pathlib import Path
import os
RESOURCES_PATH = "nlpir01/resources"
EMBEDDINGS_FILE_CT = "glove_twitter_27B_200d.txt"
EMBEDDINGS_FILE_PD = "glove_42B_300d.txt"
RESULTS_PATH = "nlpir01/results"
RESULTS_FILE_PREFIX = "T1-EN-"


PROJECT_PATH = Path(os.path.dirname(__file__))
INPUT_DATA_PATH = os.path.join(
    PROJECT_PATH.parent.parent.parent.parent.absolute(),
    'data'
)
INPUT_DATA_PATHS = {
    'covid_tweets': {
        'train': {
            'label': 'train',
            'tfpath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'train',
                'v1',
                'training.tsv',
            ),
            'v2jfpath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'train',
                'v2',
                'training_v2.json'
            )
        },
        'dev': {
            'label': 'dev',
            'tfpath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'train',
                'v1',
                'dev.tsv',
            ),
            'v2tfpath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'train',
                'v2',
                'dev_v2.tsv'
            ),
            'v2jfpath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'train',
                'v2',
                'dev_v2.json'
            )
        },
        'test': {
            'label': 'test',
            'tfpath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'test',
                'test-input.tsv',
            ),
            'jfpath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'test',
                'test-input.json',
            ),
        },
    },
    'political_debates': {
        'folderpath': os.path.join(INPUT_DATA_PATH, 'political_debates'),
        # 'embpath': os.path.join(BERT_EMB_PATH, 'political_debates'),
        'train': {
            'label': 'train',
            'folderpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'training'
            ),
            'tfpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'training',
                'train_combined.tsv'
            ),
        },
        'test_no_annotation': {
            'label': 'test_no_annotation',
            'folderpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'test_no_annotation'
            ),
            'tfpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'test_no_annotation',
                'test_no_annotation_combined.tsv'
            )
        },
        'test': {
            'label': 'test',
            'folderpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'test'
            ),
            'tfpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'test',
                'test_combined.tsv'
            )
        },
        'dev': os.path.join(
            INPUT_DATA_PATH,
            'political_debates',
            'val_combined.tsv'
        )
    }
}

TRAINING_TWEETS_PATH = "data/training_v2.json"
TRAINING_PATH = "data/training_v2.tsv"

DEV_TWEETS_PATH = "data/dev_v2.json"
DEV_PATH = "data/dev_v2.tsv"

TEST_TWEETS_PATH = "test-input/test-input.json"
TEST_PATH = "test-input/test-input.tsv"


RESULTS_PER_CLAIM = 0

COL_NAMES = {
    'covid_tweets': {
        'textc': 'tweet_text',
        'labelc': 'check_worthiness',
    },
    'political_debates': {
        'textc': 'content',
        'labelc': 'worthy',
    }
}

TEXT_COLUMN = "tweet_text"
# -v1
# LABEL_COLUMN = "claim_worthiness"
# +v1
LABEL_COLUMN = "check_worthiness"

SEQ_LEN = 50

NUM_WORDS = 15000
EMBEDDING_SIZE = 200

BATCH_SIZE = 32
EPOCHS = 100
