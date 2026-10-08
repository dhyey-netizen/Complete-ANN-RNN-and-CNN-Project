# Easy Deep Learning Projects: ANN, RNN and CNN

Three beginner-friendly end-to-end deep learning projects. All three use datasets
that download automatically, so no manual data setup is needed.

| Project | Folder | Task | Model |
| --- | --- | --- | --- |
| ANN | `ann_iris/` | Classify iris flowers into 3 species from 4 measurements | Dense(16) - Dense(8) - Dense(3, softmax) |
| RNN | `rnn_imdb/` | Classify IMDB movie reviews as positive or negative | Embedding - SimpleRNN / LSTM - Dense(1, sigmoid) |
| CNN | `cnn_mnist/` | Recognize handwritten digits 0-9 from 28x28 images | Conv32 - Pool - Conv64 - Pool - Flatten - Dropout - Dense(10, softmax) |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow supports Python 3.9 to 3.12. You can also run the scripts in Google Colab.

## Run

```bash
cd ann_iris && python iris_ann.py
cd ../rnn_imdb && python imdb_rnn.py
cd ../cnn_mnist && python cnn_mnist.py
```

Each script prints its metrics and saves plots (`learning_curves.png`,
`correlation_heatmap.png`, `rnn_vs_lstm.png`, `confusion_matrix.png`, `sample_digits.png`, `mistakes.png`) and a trained `.keras` model
in its own folder.

## What each project covers

**ANN (Iris)**: loading data, EDA, stratified train/test split, feature scaling,
building a Keras Sequential network, training with a validation split,
confusion matrix and classification report, learning curves, saving the model
and predicting a new sample.

**RNN (IMDB)**: sequence data, a fixed vocabulary, padding, an Embedding layer,
SimpleRNN versus LSTM, vanishing gradients, evaluation on 25,000 test reviews,
and predicting your own sentences.

**CNN (MNIST)**: image data as pixel arrays, scaling pixels to 0-1, convolution and pooling layers, flattening, dropout, early stopping, confusion matrix on 10 classes, and viewing the images the model gets wrong.

## Results

Fill in after running (numbers vary slightly from run to run):

| Model | Test accuracy |
| --- | --- |
| ANN (Iris) | |
| SimpleRNN (IMDB) | |
| LSTM (IMDB) | |
| CNN (MNIST) | |

## Experiments to try

- ANN: change neurons per layer, number of layers, add `Dropout(0.2)`, change the learning rate, add `EarlyStopping`.
- RNN: try `GRU`, `Bidirectional(LSTM(32))`, `MAXLEN` of 100/200/400, embedding sizes of 16/32/64.
- CNN: change filters (16/32/64), kernel size (3 or 5), add a third conv block, change dropout, or switch to Fashion-MNIST (`keras.datasets.fashion_mnist`) for a harder task.
