# 🍋Lemon🍋

> **基于numpy的基本机器学习算法库**
> 
> 算法介绍详见个人博客[《机器学不动了》专栏](https://riroaki.github.io/categories/机器学不动了/)

## Structure

### 有监督Supervised

- 线性类
  - 线性回归（基于梯度/基于normal equation）
  - 逻辑回归分类
  - 感知机分类
  - SVM分类
- 非线性类
  - 贝叶斯分类
  - k近邻分类
  - 决策树分类
- 其他
  - Regularizer正则项（L1/L2）
  - MultiClassifier多分类

### 无监督Unsupervised

- K-Means聚类
- Spectral聚类
- PCA主成分分析

### 聚合Emsemble

- Random Forest
- Adaboost

### 神经网络NN

- TODO

### 工具Utils

- batch批量分割
- scaling缩放（min-max/mean/standardizatioin/unit）
- cross validation交叉验证（K-fold/Leave-one-out）

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
  - [x] Decision Tree
- 7.2
  - [x] Multi-classifier
  - [x] Regularization
  - [x] Random Forest
  - [x] Adaboost

## TODO️

- [ ] 算法可视化
- [ ] 补充测试
