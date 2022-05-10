Advanced examples
=================

Here is a summary of our results. Remark that from one seed to the other, results of the different notebooks may vary. Please look in the different notebooks for details.

.. list-table::
    :widths: 10 15 10 10 10 10
    :header-rows: 1
    :stub-columns: 1

    * - List table
      - Dataset
      - Metric
      - Clear
      - Quantized
      - FHE
    * - Linear Regression
      - Synthetic 1D
      - R2
      - 0.876
      - 0.863
      - 0.863
    * - Logistic Regression
      - Synthetic 2D with 2 classes
      - accuracy
      - 0.90
      - 0.875
      - 0.875
    * - Poisson Regression
      - `OpenML insurance (freq) <https://www.openml.org/d/41214>`_
      - mean Poisson deviance
      - 0.61
      - 0.60
      - 0.60
    * - Gamma Regression
      - `OpenML insurance (freq) <https://www.openml.org/d/41214>`__ `OpenML insurance (sev) <https://www.openml.org/d/41215>`__
      - mean Gamma deviance
      - 0.45
      - 0.45
      - 0.45
    * - Tweedie Regression
      - `OpenML insurance (freq) <https://www.openml.org/d/41214>`__ `OpenML insurance (sev) <https://www.openml.org/d/41215>`__
      - mean Tweedie deviance (power=1.9) 
      - 33.42
      - 34.18
      - 34.18
    * - Decision Tree
      - `OpenML spams <https://www.openml.org/d/44>`_
      - precision score
      - 0.95
      - 0.97
      - 0.97\*
    * - XGBoost
      - `Diabetes <https://www.openml.org/d/37>`_
      - MCC
      - 0.48
      - 0.52
      - 0.52\*
    * - Fully Connected
      - `Iris <https://www.openml.org/d/61>`_
      - accuracy
      - 0.947
      - 0.895
      - 0.895
    * - CNN
      - `Digits <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html>`_
      - accuracy
      - 0.90
      - \*\*
      - \*\*

In this table, \* means that accuracy in FHE is assumed to be the same than the one in quantized, by just checking that 10 samples behave the same way between FHE and quantized (because things are too slow to be exhaustively launched on the full dataset). \*\* means that the accuracy is actually random-like, because the quantization we need to set to fullfill bitsize constraints are too strong.


Scikit-learn
**********************

.. toctree::
   :maxdepth: 1

   LinearRegression.ipynb
   LogisticRegression.ipynb
   PoissonRegression.ipynb
   DecisionTreeClassifier.ipynb
   XGBClassifier.ipynb
   GLMComparison.ipynb

Torch
**********************

.. toctree::
   :maxdepth: 1

   FullyConnectedNeuralNetwork.ipynb
   ConvolutionalNeuralNetwork.ipynb

Comparison of Classifiers
**************************

.. toctree::
   :maxdepth: 1

   ClassifierComparison.ipynb

Kaggle Competition
**************************

.. toctree::
   :maxdepth: 1

   KaggleTitanic.ipynb

