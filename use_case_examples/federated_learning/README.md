# Federated Learning

This use case example combines both federated learning for training and Fully Homomorphic Encryption (FHE) for inference on encrypted data. This approach allows both training and inference in the privacy preserving settings.
This example is inspired from the [flower blog post on LogisticRegression training using federated learning with flower](https://flower.dev/blog/2021-07-21-federated-scikit-learn-using-flower/).
[Federated learning files source.](https://github.com/adap/flower/tree/9ee473152f2fde2a5b05b99829db27a607dc46ec/examples/sklearn-logreg-mnist)

## Setup

To run this example you will need to install the requirements from `requirements.txt`, using `python -m pip install -r requirements.txt`.
If you are using macOS with arm then you might need to launch the `./fix_grpc.sh` script for fix the `grpcio` package installation.

## Train the model

To train the model with federated learning just use the `run.sh` script.
Once this is done you have a pickled `LogisticRegression` model that was trained using federated learning with 2 clients.

## Using the model with Concrete ML

You can then launch `python load_to_cml.py` script to load the model, compile it to FHE and then evaluate the loaded model on the test set.
