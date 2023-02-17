import os

os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

import time

import numpy
import onnx
import pandas as pd
import torch
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import text_to_tensor

from concrete.ml.sklearn import XGBClassifier


def train():
    # Get data
    if not os.path.isfile("local_datasets/twitter-airline-sentiment/Tweets.csv"):
        raise ValueError(
            "Please launch the `download_data.sh` script in order to get the datasets."
        )

    train = pd.read_csv("local_datasets/twitter-airline-sentiment/Tweets.csv", index_col=0)
    text_X = train["text"]
    y = train["airline_sentiment"]
    y = y.replace(["negative", "neutral", "positive"], [0, 1, 2])

    pos_ratio = y.value_counts()[2] / y.value_counts().sum()
    neg_ratio = y.value_counts()[0] / y.value_counts().sum()
    neutral_ratio = y.value_counts()[1] / y.value_counts().sum()

    print(f"Proportion of positive examples: {round(pos_ratio * 100, 2)}%")
    print(f"Proportion of negative examples: {round(neg_ratio * 100, 2)}%")
    print(f"Proportion of neutral examples: {round(neutral_ratio * 100, 2)}%")

    # Split in train test
    text_X_train, text_X_test, y_train, y_test = train_test_split(
        text_X, y, test_size=0.1, random_state=42
    )

    # Let's first build a representation vector from the text
    tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X_train = tfidf_vectorizer.fit_transform(text_X_train)
    X_test = tfidf_vectorizer.transform(text_X_test)

    # Make our train and test dense array
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # Let's build our model
    model = XGBClassifier()

    # A gridsearch to find the best parameters
    parameters = {
        "n_bits": [2, 3],
        "max_depth": [1],
        "n_estimators": [10, 30, 50],
        "n_jobs": [-1],
    }

    # Run the gridsearch
    grid_search = GridSearchCV(model, parameters, cv=3, n_jobs=1, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    # Check the accuracy of the best model
    print(f"Best score: {grid_search.best_score_}")

    # Check best hyper-parameters
    print(f"Best parameters: {grid_search.best_params_}")

    # Extract best model
    best_model = grid_search.best_estimator_

    # Compute the average precision for each class
    y_proba_test_tfidf = best_model.predict_proba(X_test)

    # Compute accuracy
    y_pred_test_tfidf = numpy.argmax(y_proba_test_tfidf, axis=1)
    accuracy_tfidf = numpy.mean(y_pred_test_tfidf == y_test)
    print(f"Accuracy: {accuracy_tfidf:.4f}")

    y_pred_positive = y_proba_test_tfidf[:, 2]
    y_pred_negative = y_proba_test_tfidf[:, 0]
    y_pred_neutral = y_proba_test_tfidf[:, 1]

    ap_positive_tfidf = average_precision_score((y_test == 2), y_pred_positive)
    ap_negative_tfidf = average_precision_score((y_test == 0), y_pred_negative)
    ap_neutral_tfidf = average_precision_score((y_test == 1), y_pred_neutral)

    print(f"Average precision score for positive class: " f"{ap_positive_tfidf:.4f}")
    print(f"Average precision score for negative class: " f"{ap_negative_tfidf:.4f}")
    print(f"Average precision score for neutral class: " f"{ap_neutral_tfidf:.4f}")

    # Let's see what are the top predictions based on the probabilities in y_pred_test
    print("5 most positive tweets (class 2):")
    for i in range(5):
        print(text_X_test.iloc[y_proba_test_tfidf[:, 2].argsort()[-1 - i]])

    print("-" * 100)

    print("5 most negative tweets (class 0):")
    for i in range(5):
        print(text_X_test.iloc[y_proba_test_tfidf[:, 0].argsort()[-1 - i]])

    # Compile the model to get the FHE inference engine
    # (this may take a few minutes depending on the selected model)
    start = time.perf_counter()
    best_model.compile(X_train)
    end = time.perf_counter()
    print(f"Compilation time: {end - start:.4f} seconds")

    # Let's write a custom example and predict in FHE
    tested_tweet = ["AirFrance is awesome, almost as much as Zama!"]
    X_tested_tweet = tfidf_vectorizer.transform(numpy.array(tested_tweet)).toarray()
    clear_proba = best_model.predict_proba(X_tested_tweet)

    # Now let's predict with FHE over a single tweet and print the time it takes
    start = time.perf_counter()
    decrypted_proba = best_model.predict_proba(X_tested_tweet, execute_in_fhe=True)
    end = time.perf_counter()
    print(f"FHE inference time: {end - start:.4f} seconds")

    # In[16]:

    print(f"Probabilities from the FHE inference: {decrypted_proba}")
    print(f"Probabilities from the clear model: {clear_proba}")

    # To sum up,
    # - We trained a XGBoost model over TF-IDF representation of the tweets and their respective sentiment class.
    # - The grid search gives us a model that achieves around ~70% accuracy.
    # - Given the imbalance in the classes, we rather compute the average precision per class.
    #
    # Now we will see how we can approach the problem by leveraging the transformers power.

    # ### 2. A transformer approach to text representation
    #
    # [**Transformers**](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) are neural networks that are often trained to predict the next words to appear in a text (this is commonly called self-supervised learning).
    #
    # They are powerful tools for all kind of Natural Language Processing tasks but supporting a transformer model in FHE might not always be ideal as they are quite big models. However, we can still leverage their hidden representation for any text and feed it to a more FHE friendly machine learning model (in this notebook we will use XGBoost) for classification.
    #
    # Here we will use the transformer model from the amazing [**Huggingface**](https://huggingface.co/) repository.

    # In[17]:

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the tokenizer (converts text to tokens)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    # Load the pre-trained model
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    # In[18]:

    # Let's first see what are the model performance by itself
    list_text_X_test = text_X_test.tolist()

    tokenized_text_X_test = tokenizer.batch_encode_plus(
        list_text_X_test, pad_to_max_length=True, return_tensors="pt"
    )["input_ids"]

    # Depending on the hardware used, the number of examples to be processed can be reduced
    # Here we split the data into 100 examples per batch
    tokenized_text_X_test_split = torch.split(tokenized_text_X_test, split_size_or_sections=50)
    transformer_model = transformer_model.to(device)

    outputs = []
    for tokenized_x_test in tqdm.tqdm(tokenized_text_X_test_split):
        tokenized_x = tokenized_x_test.to(device)
        output_batch = transformer_model(tokenized_x)["logits"]
        output_batch = output_batch.detach().cpu().numpy()
        outputs.append(output_batch)

    outputs = numpy.concatenate(outputs, axis=0)

    # In[19]:

    # Let's see what the transformer model predicts
    print(f"Predictions for the first 3 tweets:\n {outputs[:3]}")

    # In[20]:

    # Compute the metrics for each class

    # Compute accuracy
    accuracy_transformer_only = numpy.mean(numpy.argmax(outputs, axis=1) == y_test)
    print(f"Accuracy: {accuracy_transformer_only:.4f}")

    y_pred_positive = outputs[:, 2]
    y_pred_negative = outputs[:, 0]
    y_pred_neutral = outputs[:, 1]

    ap_positive_transformer_only = average_precision_score((y_test == 2), y_pred_positive)
    ap_negative_transformer_only = average_precision_score((y_test == 0), y_pred_negative)
    ap_neutral_transformer_only = average_precision_score((y_test == 1), y_pred_neutral)

    print(f"Average precision score for positive class: " f"{ap_positive_transformer_only:.4f}")
    print(f"Average precision score for negative class: " f"{ap_negative_transformer_only:.4f}")
    print(f"Average precision score for neutral class: " f"{ap_neutral_transformer_only:.4f}")

    # It looks like the transformer outperforms the model built on TF-IDF reprensentation.
    # Unfortunately, running a transformer that big in FHE would be highly inefficient.
    #
    # Let's see if we can leverage transformer representation and train a FHE model for the given classification task.

    # In[21]:

    # Let's vectorize the text using the transformer
    list_text_X_train = text_X_train.tolist()
    list_text_X_test = text_X_test.tolist()

    X_train_transformer = text_to_tensor(list_text_X_train, transformer_model, tokenizer, device)
    X_test_transformer = text_to_tensor(list_text_X_test, transformer_model, tokenizer, device)

    # Now we have a representation for each tweet, we can train a model on these.
    grid_search = GridSearchCV(model, parameters, cv=3, n_jobs=1, scoring="accuracy")
    grid_search.fit(X_train_transformer, y_train)

    # In[23]:

    # Check the accuracy of the best model
    print(f"Best score: {grid_search.best_score_}")

    # Check best hyper-parameters
    print(f"Best parameters: {grid_search.best_params_}")

    # Extract best model
    best_model = grid_search.best_estimator_

    # Compute the metrics for each class

    y_proba = best_model.predict_proba(X_test_transformer)

    # Compute the accuracy
    y_pred = numpy.argmax(y_proba, axis=1)
    accuracy_transformer_xgboost = numpy.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy_transformer_xgboost:.4f}")

    y_pred_positive = y_proba[:, 2]
    y_pred_negative = y_proba[:, 0]
    y_pred_neutral = y_proba[:, 1]

    ap_positive_transformer_xgboost = average_precision_score((y_test == 2), y_pred_positive)
    ap_negative_transformer_xgboost = average_precision_score((y_test == 0), y_pred_negative)
    ap_neutral_transformer_xgboost = average_precision_score((y_test == 1), y_pred_neutral)

    print(f"Average precision score for positive class: " f"{ap_positive_transformer_xgboost:.4f}")
    print(f"Average precision score for negative class: " f"{ap_negative_transformer_xgboost:.4f}")
    print(f"Average precision score for neutral class: " f"{ap_neutral_transformer_xgboost:.4f}")

    # Our FHE-friendly XGBoost model does 38% better than the XGBoost model built over TF-IDF representation of the text. Note that here we are still not using FHE and only evaluating the model.
    # Interestingly, using XGBoost over the transformer representation of the text matches the performance of the transformer model alone.

    # In[26]:

    # Get probabilities predictions in clear
    y_pred_test = best_model.predict_proba(X_test_transformer)

    # Let's see what are the top predictions based on the probabilities in y_pred_test
    print("5 most positive tweets (class 2):")
    for i in range(5):
        print(text_X_test.iloc[y_pred_test[:, 2].argsort()[-1 - i]])

    print("-" * 100)

    print("5 most negative tweets (class 0):")
    for i in range(5):
        print(text_X_test.iloc[y_pred_test[:, 0].argsort()[-1 - i]])

    # In[27]:

    # Now let's see where the model is wrong
    y_pred_test_0 = y_pred_test[y_test == 0]
    text_X_test_0 = text_X_test[y_test == 0]

    print("5 most positive (predicted) tweets that are actually negative (ground truth class 0):")
    for i in range(5):
        print(text_X_test_0.iloc[y_pred_test_0[:, 2].argsort()[-1 - i]])

    print("-" * 100)

    y_pred_test_2 = y_pred_test[y_test == 2]
    text_X_test_2 = text_X_test[y_test == 2]
    print("5 most negative (predicted) tweets that are actually positive (ground truth class 2):")
    for i in range(5):
        print(text_X_test_2.iloc[y_pred_test_2[:, 0].argsort()[-1 - i]])

    # Interestingly, these misclassifications are not obvious and some actually look rather like mislabeled. Also, it seems that the model is having a hard time to find ironic tweets.
    #
    # Now we have our model trained which has some great accuracy. Let's have it predict over the encrypted representation.

    # ### Sentiment Analysis of the Tweet with Fully Homomorphic Encryption
    #
    # Now that we have our model ready for FHE inference and our data ready for encryption let's use the model in a privacy preserving manner with FHE.

    # In[28]:

    # Compile the model to get the FHE inference engine
    # (this may take a few minutes depending on the selected model)
    start = time.perf_counter()
    best_model.compile(X_train_transformer)
    end = time.perf_counter()
    print(f"Compilation time: {end - start:.4f} seconds")

    # Let's write a custom example and predict in FHE
    tested_tweet = ["AirFrance is awesome, almost as much as Zama!"]
    X_tested_tweet = text_to_tensor(tested_tweet, transformer_model, tokenizer, device)
    clear_proba = best_model.predict_proba(X_tested_tweet)

    # Now let's predict with FHE over a single tweet and print the time it takes
    start = time.perf_counter()
    decrypted_proba = best_model.predict_proba(X_tested_tweet, execute_in_fhe=True)
    end = time.perf_counter()
    fhe_exec_time = end - start
    print(f"FHE inference time: {fhe_exec_time:.4f} seconds")

    # In[29]:

    print(f"Probabilities from the FHE inference: {decrypted_proba}")
    print(f"Probabilities from the clear model: {clear_proba}")

    # In[ ]:

    # Let's export the final model such that we can reuse it in a client/server environment

    # Export the model to ONNX
    onnx.save(best_model._onnx_model_, "server_model.onnx")  # pylint: disable=protected-access

    # Export some data to be used for compilation
    X_train_numpy = X_train_transformer[:100]

    # Merge the two arrays in a pandas dataframe
    X_test_numpy_df = pd.DataFrame(X_train_numpy)

    # to csv
    X_test_numpy_df.to_csv("samples_for_compilation.csv")

    # Let's save the model to be pushed to a server later
    from concrete.ml.deployment import FHEModelDev

    fhe_api = FHEModelDev("./dev", best_model)
    fhe_api.save()


if __name__ == "__main__":
    train()
