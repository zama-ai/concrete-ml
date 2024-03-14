# See all tutorials

## Start here

- [Build-in model examples](ml_examples.md)
- [Deep learning examples](dl_examples.md)

## Go further

### Live demos on Hugging Face:

- [Credit card approval](https://huggingface.co/spaces/zama-fhe/credit_card_approval_prediction): Predicting credit scoring card approval application in which sensitive data can be shared and analyzed without exposing the actual information to neither the three parties involved, nor the server processing it.
  - Check the code [here](https://huggingface.co/spaces/zama-fhe/credit_card_approval_prediction/tree/main)
- [Sentiment analysis with transformers](https://huggingface.co/blog/sentiment-analysis-fhe): predicting if an encrypted tweet / short message is positive, negative or neutral, using FHE.
  - Check the code [here](https://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis/tree/main) and the [blog post](https://huggingface.co/blog/sentiment-analysis-fhe)
- [Health diagnosis](https://huggingface.co/spaces/zama-fhe/encrypted_health_prediction): giving a diagnosis using FHE to preserve the privacy of the patient based on a patient's symptoms, history and other health factors.
  - Check the code [here](https://huggingface.co/spaces/zama-fhe/encrypted_health_prediction/tree/main)
- [Encrypted image filtering](https://huggingface.co/spaces/zama-fhe/encrypted_image_filtering): filtering encrypted images by applying filters such as black-and-white, ridge detection, or your own filter.
  - Check the code [here](https://huggingface.co/spaces/zama-fhe/encrypted_image_filtering/tree/main)

### Code examples on Github:

- [GPT-2 in FHE](../../use_case_examples/llm/README.md): Privacy-preserving text generation based on a user's prompt
- [Titanic](../../use_case_examples/titanic/README.md): Train an XGB classifier that can perform encrypted prediction for the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/)
- [Federated learning and private inference](../../use_case_examples/federated_learning/README.md): Use federated learning to train a Logistic Regression while preserving training data confidentiality. Import the model into Concrete ML and perform encrypted prediction
- [Neutral network fine-tuning](../../use_case_examples/cifar/cifar_brevitas_finetuning/README.md): Fine-tune a VGG network to classify the CIFAR image data-sets and predict on encrypted data
- [Encrypted sentiment analysis](../../use_case_examples/sentiment_analysis_with_transformer/README.md):A Hugging Face space that securely analyzes the sentiment expressed in a short text
- [Credit scoring](../../use_case_examples/credit_scoring/README.md): Predict the chance of a given loan applicant defaulting on loan repayment

### Blog tutorials:

- [Build an end-to-end encrypted Shazam application using Concrete ML](https://www.zama.ai/post/encrypted-shazam-using-fully-homomorphic-encryption-concrete-ml-tutorial) - February 2024
- [Linear regression over encrypted data with homomorphic encryption](https://www.zama.ai/post/linear-regression-using-linear-svr-and-concrete-ml-homomorphic-encryption) - June 2023
- [Comparison of Concrete ML regressors](https://www.zama.ai/post/comparison-of-concrete-ml-regressors) - June 2023
- [How to deploy a machine learning model with Concrete ML](https://www.zama.ai/post/how-to-deploy-machine-learning-models-with-concrete-ml) - May 2023
- [Encrypted image filtering using homomorphic encryption](https://www.zama.ai/post/encrypted-image-filtering-using-homomorphic-encryption) - February 2023
- [Sentiment analysis over encrypted data](https://huggingface.co/blog/sentiment-analysis-fhe) - November 2022
- [Titanic Competition with Privacy Preserving Machine Learning](https://www.zama.ai/post/titanic-competition-with-privacy-preserving-machine-learning-using-concrete-ml) - August 2022

### Video tutorials

- [Train a linear classifier on encrypted data using Concrete ML and Fully Homomorphic Encryption (FHE)](https://www.zama.ai/post/video-tutorial-train-a-linear-classifier-on-encrypted-data-using-concrete-ml-and-fully-homomorphic-encryption-fhe) - February 2024
- [How to convert a scikit-learn model into its homomorphic equivalent](https://www.zama.ai/post/how-to-convert-a-scikit-learn-model-into-its-homomorphic-equivalent) - June 2023
