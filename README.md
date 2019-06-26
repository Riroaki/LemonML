# 🍋Lemon🍋

> **基于numpy的基本机器学习算法库**

本项目起源于ZJU2019年春夏学期的《数据挖掘导论》课程作业，代码上传在我的[另一个项目](https://github.com/Riroaki/DataMining-ZJU-19-Summer)。

近期也在个人博客的《机器学不动了》[专栏]([https://riroaki.github.io/categories/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B8%8D%E5%8A%A8%E4%BA%86/](https://riroaki.github.io/categories/机器学不动了/))同步更新项目中涉及的机器学习算法系列内容。

## 目标：

- 清晰易懂的代码和注释
- 简单易用的API
- 全面的机器学习算法

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

### 无监督Unsupervised

- K-Means聚类
- Spectral聚类
- PCA主成分分析
- ……

### 聚合Emsemble

- Bagging
  - Random Forest
- Adaboost
- ……

### 神经网络NN

- ……

### 工具Utils

- batch批量分割
- scaling缩放（min-max/mean/standardizatioin/unit）
- cross validation交叉验证（K-fold/Leave-one-out）

## API

### SupervisedModel

- Class
  - `LinearRegression`
  - `LogisticRegression`
  - `Perceptron`
  - `SVM`
  - `KNearest`
  - `DecisionTree`
  - `...`
- Methods
  - `fit(x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray`
  - `predict(x: np.ndarray, **kwargs) -> np.ndarray`
  - `evaluate(x: np.ndarray, y: np.ndarray, **kwargs) -> tuple`
  - `dump(dump_file: str) -> None`
  - `load(dump_file: str) -> None`

### UnsupervisedModel

- Class
  - `KMeans`
  - `Spectral`
  - `PCA`
  - `...`
- Methods
  - `Clustering(x: np.ndarray, **kwargs) -> np.ndarray`

### Utils

- `batch`
  - `batch(data: np.ndarray, y: np.ndarray, size: int, shuffle: bool = False) -> tuple`
- `cross_validate`
  - `k_fold(data: np.ndarray, y: np.ndarray, k: int, fit_func: callable, eval_func: callable, shuffle: bool = True) -> tuple`
  - `leave_one_out(data: np.ndarray, y: np.ndarray, fit_func: callable, eval_func: callable, shuffle: bool = True) -> tuple`
- `make_data`
  - `linear(n: int, dim: int, rand_bound: float = 10., noisy: bool = False) -> tuple`
  - `logistic(n: int, dim: int, rand_bound: float = 10., noisy: bool = False) -> tuple`
  - `perceptron(n: int, dim: int, rand_bound: float = 10., noisy: bool = False) -> tuple`
  - `svm(n: int, dim: int, rand_bound: float = 10., noisy: bool = False) -> tuple`
  - `...`
- `scaling`
  - `std(data: np.ndarray) -> None`
  - `minmax(data: np.ndarray) -> None`
  - `mean(data: np.ndarray) -> None`
  - `unit(data: np.ndarray) -> None`

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
- 6.27
  - [ ] Add multi-classifier support for binary-classifiers
  - [ ] Add Ridge and Lasso support for linear classifiers
  - [ ] Start Emsemble and Neural Network

## TODO

- [ ] 充Ridge和Lasso相关内容（预计使用装饰器）
- [ ] 目前部分分类算法为二分类，需要补充增加多分类实现
- [ ] 增加多种损失函数及对应梯度计算方法实现
- [ ] 增加boost等ensemble方法实现（基本目标：random forest）
- [ ] 对线性模型，增加多种学习率优化方式，如adagrad等

- [ ] 增加算法可视化