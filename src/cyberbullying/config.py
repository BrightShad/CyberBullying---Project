from pathlib import Path

# Project root (two levels up from this file: src/cyberbullying -> project root)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "cyberbullying_tweets.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

DATA_DIR = ROOT_DIR / "data"

GLOVE_PATH = DATA_DIR / "glove.6B.300d.txt"

def get_vocab_path(task: str) -> Path:
    return ARTIFACTS_DIR / f"vocab_{task}.json"

def get_label_map_path(task: str) -> Path:
    return ARTIFACTS_DIR / f"label_map_{task}.json"

def get_splits_path(task: str) -> Path:
    return ARTIFACTS_DIR / f"splits_{task}.pkl"

def get_processed_data_path(task: str) -> Path:
    return ARTIFACTS_DIR / f"processed_{task}.csv"

def get_tfidf_vectorizer_path(task: str) -> Path:
    return MODELS_DIR / f"tfidf_vectorizer_{task}.joblib"

def get_model_path(model_type: str, task: str) -> Path:
    if model_type == "bilstm":
        return MODELS_DIR / ("best_model_multiclass.pt" if task == "multiclass" else "best_model.pt")
    elif model_type == "logistic":
        return MODELS_DIR / f"logistic_{task}.joblib"
    elif model_type == "rf":
        return MODELS_DIR / f"rf_{task}.joblib"
    raise ValueError(f"Unknown model_type: {model_type}")

# Original dataset labels
MULTICLASS_LABELS = [
    "not_cyberbullying",
    "gender",
    "religion",
    "age",
    "ethnicity",
    "other_cyberbullying",
]

BINARY_LABELS = ["not_cyberbullying", "cyberbullying"]

# Preprocessing
MAX_SEQ_LEN = 128
MIN_WORD_FREQ = 2
MAX_VOCAB_SIZE = 20000
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# Model hyperparameters
EMBED_DIM = 300
HIDDEN_DIM = 128
DROPOUT = 0.3

# Training defaults
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
