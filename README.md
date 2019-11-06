# 🍋Lemon🍋

> **Basic Machine Learning / Deep Learning Library**
> 
> Implemented with numpy and scipy in python codes.
> 
> Also includes a simple version of autogradable Tensor.
> 
> For more information, please refer to [my blog](https://riroaki.github.io/categories/机器学不动了/).

## Requirements

- python==3.6
- numpy==1.17.0
- scipy==1.2.1
- torch==1.3.0

## Structure
```
.
├── LICENSE
├── README.md
├── graph
│   ├── __init__.py
│   ├── _conditional_random_field.py
│   └── _hidden_markov.py
├── nn
│   ├── __init__.py
│   ├── _activation.py
│   ├── _base.py
│   ├── _criterion.py
│   ├── _fully_connect.py
│   └── autograd
│       ├── __init__.py
│       └── tensor.py
├── supervised
│   ├── __init__.py
│   ├── _base.py
│   ├── bayes
│   │   ├── __init__.py
│   │   └── _bayes.py
│   ├── knn
│   │   ├── __init__.py
│   │   └── _k_nearest.py
│   ├── linear
│   │   ├── __init__.py
│   │   ├── _base.py
│   │   ├── _linear_regression.py
│   │   ├── _logistic_regression.py
│   │   ├── _multi_classifier.py
│   │   ├── _perceptron.py
│   │   ├── _regularization.py
│   │   └── _support_vector_machine.py
│   └── tree
│       ├── __init__.py
│       ├── _cart.py
│       ├── _id3.py
│       └── ensemble
│           ├── __init__.py
│           ├── _adaptive_boosting.py
│           └── _random_forest.py
├── test
│   ├── nn_models
│   │   └── fcnn.py
│   ├── test_graph.py
│   └── test_supervised.py
├── unsupervised
│   ├── __init__.py
│   ├── clustering
│   │   ├── __init__.py
│   │   ├── _base.py
│   │   ├── _kmeans.py
│   │   └── _spectral.py
│   └── decomposition
│       ├── __init__.py
│       ├── _base.py
│       └── _pca.py
└── utils
    ├── __init__.py
    ├── _batch.py
    ├── _cross_validate.py
    ├── _make_data.py
    └── _scaling.py
```

## Timeline

- 2019.6.12
  - [x] Linear Regression
  - [x] Logistic Regression
  - [x] Perceptron
  - [x] utils.scaling / batch / cross_validate
- 6.13
  - [x] Support Vector Machine
  - [x] K-Nearest-Neighbor
  - [x] test script
- 6.15
  - [x] Bayes
- 6.16
  - [x] K-Means
- 6.19
  - [x] Spectral
  - [x] Principle Component Analysis
- 6.24
  - [x] Decision Tree(ID3)
- 7.2
  - [x] Multi-classifier
  - [x] Regularization
- 7.13
  - [x] Activation
  - [x] Criterion
  - [x] Fully Connected Layer
  - [x] Fully Connected Neural Network Model
- 8.17-8.20
  - [x] Improve project structure
  - [x] Decision Tree(CART)
  - [x] Random Forest
  - [x] Adaboost
- 8.23
  - [x] Hidden Markov Model
- 11.6
  - [x] Conditional Random Field Model(Based on `Torch`)
  - [x] Autograd Tensor