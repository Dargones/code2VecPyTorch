import logging


class Config:
    # Paths to dataset:
    DATASET = "/home/sasha/Desktop/Research/code2vec/data/Test/Test"
    TRAIN_DATA = DATASET + ".train.c2v"
    TEST_DATA = DATASET + ".test.c2v"
    VAL_DATA = DATASET + ".val.c2v"
    # Data preprocessing constants:
    MAX_CONTEXTS = 200
    MAX_TOKEN_VOCAB_SIZE = 0.95
    MAX_TARGET_VOCAB_SIZE = None
    MAX_PATH_VOCAB_SIZE = 0.9
    PROPERTIES = 0
    # Model and training constants
    SHUFFLE_BUFFER_SIZE = 10000
    BATCH_SIZE = 1024
    DEFAULT_EMBEDDINGS_SIZE = 128
    NUM_TRAIN_EPOCHS = 40
    PATIENCE_DELTA = 0.0001
    PATIENCE = 3
    LR_START = 0.01
    # Logging constants
    LOG_FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    LOG_FILE = "main.log"
    LOG_LEVEL = logging.DEBUG