# 🍋Lemon🍋

> **Basic Machine Learning / Deep Learning Library**
> 
> Implemented with numpy and scipy in python codes.
> 
> For more information, please refer to [my blog](https://riroaki.github.io/categories/机器学不动了/)

## Requirements

- python==3.6.8
- numpy==1.15.4
- scipy==1.2.1

## Structure
```
.
├── LICENSE
├── README.md
├── nn
│   ├── __init__.py
│   ├── _activation.py
│   ├── _base.py
│   ├── _criterion.py
│   └── _fully_connect.py
├── supervised
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-36.pyc
│   ├── _base.py
│   ├── bayes
│   │   ├── __init__.py
│   │   └── _bayes.py
│   ├── ensemble
│   │   ├── __init__.py
│   │   ├── _adaboost.py
│   │   ├── _multi_classifier.py
│   │   └── _random_forest.py
│   ├── knn
│   │   ├── __init__.py
│   │   └── _k_nearest.py
│   ├── linear
│   │   ├── __init__.py
│   │   ├── _base.py
│   │   ├── _linear_regression.py
│   │   ├── _logistic_regression.py
│   │   ├── _perceptron.py
│   │   ├── _regularization.py
│   │   └── _support_vector_machine.py
│   └── tree
│       ├── __init__.py
│       ├── _tree_cart.py
│       └── _tree_id3.py
├── test
│   ├── nn_models
│   │   └── fcnn.py
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
    ├── __pycache__
    │   └── __init__.cpython-36.pyc
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
- 8.17
  - [x] Improve project structure
  - [ ] Decision Tree(CART)
  - [ ] Random Forest
  - [ ] Adaboost

## TODO️

- [ ] Some codes lacks **TESTING**!!!
- [ ] Finish ensemble、nn parts...
