# Philosophy of the Design

Our primary concern in this release was the ease of adoption of our framework. That is why we built APIs, which should feel natural to data scientists. While performance is also an important concern for deployment of FHE machine learning models, improvements on this front will come in future releases.

To this end, we have decided to mimic the APIs of scikit-learn and XGBoost for machine learning models (linear models and tree-based models) and of torch for deep learning models. We refer readers to [scikit-learn how-to](../../user/howto/simple_example_sklearn.md) and to [torch how-to](../../user/howto/simple_example_torch.md), which show how similar our APIs are to their non-FHE counterparts.
