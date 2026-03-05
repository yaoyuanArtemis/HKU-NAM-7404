# Baseline Model Comparison for NAM

这个目录包含用于对比 Neural Additive Models (NAM) 与其他基线模型的代码。

## 支持的模型

1. **NAM (Neural Additive Model)** - 可解释的神经网络模型
2. **Logistic Regression / Linear Regression** - 简单的线性模型
3. **CART (Decision Tree)** - 基于树的可解释模型
4. **XGBoost** - 梯度提升树集成模型
5. **DNN-MLP** - 深度神经网络（多层感知机）
6. **EBM (Explainable Boosting Machine)** - 可解释的梯度提升模型（Boosted GAM）

## 安装依赖

首先安装基础依赖：

```bash
pip install -r requirements.txt
```

然后安装对比模型所需的额外依赖：

```bash
pip install -r requirements_baseline.txt
```

或者手动安装：

```bash
pip install xgboost scikit-learn interpret
```

## 推荐使用方法 ⭐

### **使用 `run_experiment.py` 统一对比脚本（推荐）**

这是**最简单**的方式，一个命令运行所有对比：

```bash
# 基本使用
python run_experiment.py \
    --data_path ./data/my_dataset.csv \
    --target_column label

# 完整参数示例
python run_experiment.py \
    --data_path ./data/my_dataset.csv \
    --target_column label \
    --task classification \
    --models all \
    --test_size 0.2 \
    --val_size 0.2 \
    --output_dir ./results
```

**优势：**
- ✅ 自动处理数据分割
- ✅ 自动标准化特征
- ✅ 统一的结果格式
- ✅ 自动生成对比报告（CSV + Markdown）
- ✅ 可选择运行特定模型

**输出：**
- `train_split.csv`, `val_split.csv`, `test_split.csv` - 数据分割
- `{dataset}_comparison.csv` - 结果表格
- `{dataset}_report.md` - 格式化的对比报告

---

## 其他使用方法

### 方法 1: 使用 `baseline_models.py` 模块

```python
from baseline_comparison import BaselineComparison
import numpy as np

# 准备数据
X_train, y_train = ...  # 训练数据
X_val, y_val = ...      # 验证数据
X_test, y_test = ...    # 测试数据

# 初始化对比工具
# regression=False 表示分类任务
# regression=True 表示回归任务
comparison = BaselineComparison(regression=False, random_state=42)

# 训练并评估所有模型
results_df = comparison.train_and_evaluate(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

# 查看结果
print(results_df)

# 保存结果
comparison.save_results('./results', dataset_name='my_dataset')
```

### 方法 2: 运行内置示例

```bash
python baseline_models.py
```

这会运行一个内置的示例，使用 scikit-learn 生成的数据集展示分类和回归任务。

## 输出结果

### 评估指标

- **分类任务**: AUROC (Area Under ROC Curve)
  - 值越大越好
  - 范围: [0, 1]

- **回归任务**: RMSE (Root Mean Squared Error)
  - 值越小越好
  - 范围: [0, +∞)

### 结果表格

程序会输出一个包含以下列的结果表格：

| Model | Train Metric | Val Metric | Test Metric | Train Time (s) | Num Parameters |
|-------|--------------|------------|-------------|----------------|----------------|
| ... | ... | ... | ... | ... | ... |

结果会自动按照测试集性能排序（分类任务降序，回归任务升序）。

## 集成到现有 NAM 训练流程

如果你想在 NAM 训练后自动运行基线对比：

```python
from baseline_comparison import BaselineComparison
from data_utils import load_dataset

# 1. 加载数据（使用 NAM 的数据工具）
dataset = load_dataset(...)

# 2. 训练 NAM 模型
# ... 你的 NAM 训练代码 ...

# 3. 运行基线对比
comparison = BaselineComparison(regression=FLAGS.regression)
baseline_results = comparison.train_and_evaluate(
    dataset['X_train'], dataset['y_train'],
    dataset['X_val'], dataset['y_val'],
    dataset['X_test'], dataset['y_test']
)

# 4. 对比 NAM vs Baselines
print("\nBaseline Results:")
print(baseline_results)

print("\nNAM Results:")
print(f"Test AUROC: {nam_test_auroc:.4f}")
```

## 自定义模型参数

你可以修改 `baseline_models.py` 中的 `initialize_models()` 方法来调整每个模型的超参数：

```python
def initialize_models(self):
    if self.regression:
        self.models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200,  # 增加树的数量
                max_depth=10,      # 增加树的深度
                learning_rate=0.05, # 降低学习率
                random_state=self.random_state
            ),
            # ... 其他模型 ...
        }
```

## 注意事项

1. **EBM 模型**: 如果 `interpret` 包未安装，EBM 模型会被自动跳过
2. **训练时间**: DNN-MLP 和 XGBoost 可能需要较长时间训练，特别是在大数据集上
3. **内存使用**: XGBoost 和 EBM 在大数据集上可能需要较多内存
4. **随机种子**: 使用相同的 `random_state` 确保结果可重复

## 与 NAM 论文中的对比

NAM 论文中对比了以下模型：
- Linear/Logistic Regression
- Shallow Neural Networks
- Deep Neural Networks
- Boosted Trees (XGBoost)
- EBMs (GAM²)

本代码实现了所有这些基线模型，可以复现论文中的对比实验。

## 脚本说明

| 脚本 | 类型 | 用途 | 推荐度 |
|------|------|------|--------|
| `run_experiment.py` | 命令行工具 | 一键完整对比 + NAM 指导 | ⭐⭐⭐ 最推荐 |
| `baseline_models.py` | Python 类库 | 核心模块，编程接口 | ⭐⭐⭐ 必需依赖 |
| `nam_train.py` | NAM 专用 | NAM 模型训练（项目原始文件） | ⭐⭐⭐ NAM 训练 |

## 快速开始示例

```bash
# 1. 准备你的数据（CSV 格式）
# 假设你有 data.csv，包含特征列和一个目标列 'target'

# 2. 运行完整对比
python run_experiment.py \
    --data_path data.csv \
    --target_column target \
    --task classification

# 3. 查看结果
cat comparison_results/data_comparison.csv
cat comparison_results/data_report.md
```

## 常见问题

### Q: 如何只运行某些特定模型？

```bash
# 只运行 XGBoost 和 DNN-MLP
python run_experiment.py \
    --data_path data.csv \
    --target_column target \
    --models xgboost mlp
```

### Q: 如何调整 NAM 的超参数？

```bash
python run_experiment.py \
    --data_path data.csv \
    --target_column target \
    --nam_epochs 2000 \
    --nam_lr 0.001 \
    --nam_dropout 0.3
```

### Q: 如何单独训练 NAM 模型？

NAM 有专门的训练脚本：

```bash
python nam_train.py \
    --dataset_name my_dataset \
    --regression False \
    --training_epochs 1000 \
    --learning_rate 0.01 \
    --logdir ./nam_logs
```

### Q: baseline_models.py 和 run_experiment.py 有什么区别？

- `baseline_models.py`: Python 模块，提供 `BaselineComparison` 类，用于编程接口
- `run_experiment.py`: 命令行工具，自动处理数据并运行所有对比，更方便

### Q: 我的数据很大，训练时间太长怎么办？

```bash
# 减少 NAM 训练轮数
python run_experiment.py \
    --data_path data.csv \
    --target_column target \
    --nam_epochs 500 \
    --models xgboost cart  # 只运行快速模型
```

## 参考文献

- NAM: [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912)
- EBM: [InterpretML](https://github.com/interpretml/interpret)
- XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/)
