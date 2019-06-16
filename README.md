# 🍋Lemon🍋

> **基于numpy的基本机器学习算法库**

本项目起源于ZJU2019年春夏学期的《数据挖掘导论》课程作业。

预计近期将会在[个人博客](https://riroaki.github.io/)同步更新项目中涉及的机器学习算法系列内容。

## 目标：

- 简单明了的API
- 尽可能多的机器学习内容覆盖

## Structure

```shell
➜  Lemon tree
.
├── README.md
├── supervised        ## 有监督算法
│   ├── __init__.py
│   ├── _basics.py           # Model基类，后续可以添加其他基类
│   ├── _bayes.py            # 贝叶斯分类
│   ├── _knn.py              # k近邻分类
│   ├── _linear.py           # 线性回归（基于梯度/基于normal equation）
│   ├── _logistic.py         # 逻辑回归分类
│   ├── _perceptron.py       # 感知机分类
│   ├── _svm.py              # SVM分类
│   └── _tree.py             # 决策树分类
├── test.py           ## 测试脚本：用于测试各个模块功能
├── unsupervised      ## 无监督算法
│   ├── __init__.py
│   ├── _kmeans.py           # k均值聚类
│   └── _kmedoids.py         # k中心聚类
└── utils             ## 工具函数
    ├── __init__.py
    ├── _batch.py            # 用于切分mini-batch
    ├── _cross_validate.py   # 用于交叉验证，包括k折验证与留一验证
    ├── _make_data.py        # 用于生成测试数据，检验算法正确性
    └── _scaling.py          # 用于缩放初始值，包括min-max，mean，
                             # standerdization等缩放函数
```

## API

### Supervised

- Class
  - `LinearRegression`
  - `LogisticRegression`
  - `Perceptron`
  - `SVM`
  - `KNearest`
  - `…`
- Methods
  - `fit(x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray`
  - `predict(x: np.ndarray, **kwargs) -> np.ndarray`
  - `evaluate(x: np.ndarray, y: np.ndarray, **kwargs) -> tuple`
  - `dump(dump_file: str) -> None`
  - `load(dump_file: str) -> None`

### Unsupervised

- To be continued...

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
  - `std(data: np.ndarray)`
  - `minmax(data: np.ndarray)`
  - `mean(data: np.ndarray)`
  - `unit(data: np.ndarray)`

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

- 6.17
  - [ ] Spectral
  - [ ] Decision Tree
  - [ ] PCA

## TODO

- 实现全部算法以及算法测试部分（当前目标）

- 补充Ridge和Lasso相关内容
- 对于分类算法，目前默认实现为二分类，部分分类算法需要补充增加多分类实现
- 增加多种损失函数及对应梯度计算方法实现
- 增加boost等ensemble方法实现（基本目标：random forest）
- 对线性模型，增加多种学习率优化方式，如adagrad等

- 增加算法可视化……