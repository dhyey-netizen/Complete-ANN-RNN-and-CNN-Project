Deep Learning Projects: ANN, RNN and CNN

Three beginner-friendly, end-to-end deep learning projects built with Python, TensorFlow/Keras and scikit-learn. Each project covers the full workflow: data loading, preprocessing, model design, training, validation, evaluation and inference. The ANN is also served as a REST API with FastAPI and Docker.

All three projects use datasets that download automatically, so no manual data setup is needed.
Projects
Project	Folder	Dataset	Task	Model
ANN	ann_iris/	Iris (150 samples, 4 features)	Classify flowers into 3 species	Dense 16 - Dense 8 - Dense 3 (softmax)
RNN	rnn_imdb/	IMDB reviews (50,000 reviews)	Classify reviews as positive or negative	Embedding - SimpleRNN / LSTM - Dense 1 (sigmoid)
CNN	cnn_mnist/	MNIST (70,000 images, 28x28)	Recognise handwritten digits 0-9	Conv 32 - Pool - Conv 64 - Pool - Flatten - Dropout - Dense 10 (softmax)
API	deploy_api/	Trained Iris ANN	Serve predictions over HTTP	FastAPI + Docker
What each project covers

ANN (Iris)

Exploratory data analysis and a feature correlation heatmap
Stratified train/test split and feature scaling (scaler fitted on training data only)
Keras Sequential network with ReLU hidden layers and a softmax output
Training with a validation split, learning curves, confusion matrix and classification report
Hyperparameter experiments: neurons, layers, dropout, learning rate, early stopping

RNN (IMDB)

Sequence data: a fixed 10,000-word vocabulary and padding to 200 words
Embedding layer feeding a recurrent layer
SimpleRNN compared with LSTM, showing the effect of vanishing gradients
Evaluation on 25,000 unseen test reviews and predictions on custom sentences

CNN (MNIST)

Image data as pixel arrays, scaled to 0-1
Two convolution + max-pooling blocks, Flatten, Dropout
Early stopping, a 10x10 confusion matrix and a view of the misclassified images

Setup
bash
git clone "link of this repo"
cd deep-learning-projects

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

TensorFlow supports Python 3.9 to 3.12. You can also run the scripts in Google Colab.

Run
bash
cd ann_iris && python iris_ann.py
cd ../rnn_imdb && python imdb_rnn.py
cd ../cnn_mnist && python cnn_mnist.py

Each script prints its metrics, saves its plots (learning curves, confusion matrices and so on) and saves a trained .keras model in its own folder.

Results

Test-set results from my runs (numbers vary slightly between runs):

Model	Test accuracy
ANN (Iris)	fill in
SimpleRNN (IMDB)	fill in
LSTM (IMDB)	fill in
CNN (MNIST)	fill in
Deploy the ANN as an API

After running ann_iris/iris_ann.py, copy iris_ann.keras and iris_scaler.joblib into deploy_api/.

bash
cd deploy_api
pip install -r requirements.txt
uvicorn app:app --reload

Open http://127.0.0.1:8000/docs, or call it directly:

bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

To run it in Docker or put it online, see deploy_api/DEPLOY.md.

Tech stack

Python, TensorFlow / Keras, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, FastAPI, Uvicorn, Docker, Git.

Ideas for next steps
Tune each model with a systematic search (for example Keras Tuner)
Try GRU and Bidirectional LSTM on the IMDB data
Use data augmentation or transfer learning for image tasks
Serve the RNN and CNN models through the API as well
Add automated tests and a CI workflow

Author
Dhyey Gajjar
