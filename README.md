#********** Cyberbullying Detection *********#

A machine learning project designed to detect and classify cyberbullying in tweets. The pipeline is built to handle both binary classification (detecting if a tweet is bullying or not) and multi-class classification (categorizing the type of bullying, such as gender, religion, age, or ethnicity).

## How it works

The project is split into two main modeling approaches:

1.Deep Learning (PyTorch): A Bidirectional LSTM (BiLSTM) neural network. To help the model understand the English language better on a smaller dataset, we injected Stanford's pre-trained GloVe embeddings (300-dimensional vectors) directly into the embedding layer.
2.Traditional ML (Scikit-Learn): Logistic Regression and Random Forest models trained on TF-IDF vectors. These act as our baseline models to compare against the neural network.

The entire data pipeline is built from scratch. A preprocessing script cleans the text (removing URLs, mentions, and symbols), builds a custom vocabulary dictionary, and creates perfectly stratified train/validation/test splits so the models evaluate fairly.

## Running the code

Before training any models, you need to process the data and download the GloVe embeddings.

Process the dataset for either binary or multiclass:
`python scripts/preprocess.py --task binary`
`python scripts/preprocess.py --task multiclass`

Download the GloVe embeddings:
`python scripts/download_glove.py`

Once the data is processed, you can train and evaluate the models:
`python scripts/train.py --task binary`
`python scripts/evaluate.py --task binary`

To train the machine learning baselines (Logistic Regression and Random Forest):
`python scripts/train_baseline.py --task binary`

(You can swap `--task binary` for `--task multiclass` in any of the above commands).

## API

The project includes a FastAPI server that loads all trained models into memory for fast inference. 
Start it by running:
`uvicorn api.main:app --reload`
You can then send POST requests to `http://localhost:8000/predict` with your text to get live predictions.
