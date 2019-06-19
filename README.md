# 🍋Lemon🍋

> **基于numpy的基本机器学习算法库**

本项目起源于ZJU2019年春夏学期的《数据挖掘导论》课程作业。

预计近期将会在[个人博客](https://riroaki.github.io/)同步更新项目中涉及的机器学习算法系列内容。

## 目标：

- 清晰易懂的代码和注释
- 简单易用的API
- 全面的机器学习算法

## Structure

### 有监督类

- 线性类
  - 线性回归（基于梯度/基于normal equation）
  - 逻辑回归分类
  - 感知机分类
  - SVM分类
- 非线性类
  - 贝叶斯分类
  - k近邻分类
  - 决策树分类

### 无监督类

- K-Means聚类
- Spectral聚类
- PCA主成分分析
- ……

### 工具函数

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
  - [ ] Spectral
  - [x] Principle Component Analysis
  - [ ] Decision Tree

## TODO

- 实现全部算法以及算法测试部分（当前目标）

- 补充Ridge和Lasso相关内容
- 对于分类算法，目前默认实现为二分类，部分分类算法需要补充增加多分类实现
- 增加多种损失函数及对应梯度计算方法实现
- 增加boost等ensemble方法实现（基本目标：random forest）
- 对线性模型，增加多种学习率优化方式，如adagrad等

- 增加算法可视化