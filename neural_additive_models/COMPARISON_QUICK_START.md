# 模型对比快速开始指南

## 🎯 最简单的方式（推荐）

```bash
python compare_all_models.py \
    --data_path your_data.csv \
    --target_column your_target_column
```

就这么简单！脚本会自动：
- ✅ 分割数据（训练/验证/测试）
- ✅ 标准化特征
- ✅ 训练所有基线模型
- ✅ 生成对比报告

## 📁 对比脚本说明（仅2个文件）

### 1️⃣ `compare_all_models.py` ⭐ **命令行工具（推荐）**
**一键运行完整对比**

**功能**:
- ✅ 自动数据预处理和分割
- ✅ 支持选择特定模型运行
- ✅ 自动生成格式化报告（CSV + Markdown）
- ✅ 提供 NAM 训练指导
- ✅ 保存数据分割和配置

**使用**:
```bash
# 基本用法：运行所有模型
python compare_all_models.py \
    --data_path data.csv \
    --target_column label

# 高级用法：只运行部分模型
python compare_all_models.py \
    --data_path data.csv \
    --target_column label \
    --models xgboost cart mlp

# 自定义参数
python compare_all_models.py \
    --data_path data.csv \
    --target_column label \
    --task regression \
    --test_size 0.15 \
    --output_dir ./my_results
```

---

### 2️⃣ `baseline_comparison.py` - **Python 编程接口**
**给开发者的核心类库**

**功能**:
- 💻 提供 `BaselineComparison` 类
- 🔧 适合集成到自定义代码
- 🎛️ 灵活控制训练流程
- 📦 被 `compare_all_models.py` 调用

**使用**:
```python
from baseline_comparison import BaselineComparison

# 准备数据
X_train, y_train = ...
X_val, y_val = ...
X_test, y_test = ...

# 运行对比
comparison = BaselineComparison(regression=False)
results = comparison.train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test
)
print(results)
```

---

### 💡 NAM 模型训练

NAM 有独立的训练脚本 `nam_train.py`（项目原始文件）

```bash
python nam_train.py \
    --dataset_name my_dataset \
    --regression False \
    --training_epochs 1000 \
    --logdir ./nam_logs
```

---

## 🚀 快速对比流程

### 步骤 1: 准备数据
确保你的数据是 CSV 格式，包含特征列和一个目标列。

### 步骤 2: 运行对比
```bash
python compare_all_models.py \
    --data_path your_data.csv \
    --target_column target
```

### 步骤 3: 查看结果
```bash
# 查看 CSV 结果
cat comparison_results/your_data_comparison.csv

# 查看 Markdown 报告
cat comparison_results/your_data_report.md
```

---

## 📊 结果示例

运行后会得到类似这样的对比表格：

| Model | Train AUROC | Val AUROC | Test AUROC | Train Time (s) | Num Parameters |
|-------|-------------|-----------|------------|----------------|----------------|
| XGBoost | 0.9823 | 0.9456 | 0.9412 | 2.34 | 3100 |
| EBM | 0.9654 | 0.9401 | 0.9388 | 8.91 | 2000 |
| DNN-MLP | 0.9891 | 0.9378 | 0.9345 | 12.45 | 8321 |
| CART | 0.9234 | 0.8891 | 0.8876 | 0.12 | 127 |
| Logistic | 0.8745 | 0.8523 | 0.8501 | 0.08 | 21 |

---

## 🔧 常用参数

### 任务类型
```bash
--task classification  # 分类任务（默认）
--task regression      # 回归任务
```

### 模型选择
```bash
--models all                    # 所有模型（默认）
--models xgboost cart           # 只运行 XGBoost 和 CART
--models logistic xgboost mlp   # 只运行这三个
```

可选模型：
- `logistic` / `linear` - 逻辑回归/线性回归
- `cart` - 决策树
- `xgboost` - XGBoost
- `mlp` - DNN-MLP
- `ebm` - 可解释提升机
- `nam` - NAM（需要额外配置）

### 数据分割
```bash
--test_size 0.2   # 测试集比例（默认 20%）
--val_size 0.2    # 验证集比例（默认 20%）
```

### 输出目录
```bash
--output_dir ./results  # 指定结果保存目录
```

---

## 💡 使用建议

### 场景 1: 快速对比所有基线模型（命令行）
```bash
python compare_all_models.py \
    --data_path data.csv \
    --target_column label
```
⭐ **推荐**：自动生成完整报告

---

### 场景 2: 只对比特定模型（如 XGBoost 和树模型）
```bash
python compare_all_models.py \
    --data_path data.csv \
    --target_column label \
    --models xgboost cart
```
🎯 **针对性对比**：节省时间

---

### 场景 3: 在代码中集成对比功能
```python
from baseline_comparison import BaselineComparison

comparison = BaselineComparison(regression=False)
results = comparison.train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test
)
```
💻 **编程接口**：灵活控制

---

## ❓ 常见问题

**Q: 为什么 NAM 没有直接集成到对比脚本中？**

A: NAM 使用 TensorFlow 1.x，有独立的训练流程。`compare_all_models.py` 会告诉你如何单独运行 NAM，并提供命令示例。

**Q: 哪个脚本最适合我？**

- **命令行快速对比** → `compare_all_models.py` ⭐ (推荐)
- **编程接口集成** → `baseline_comparison.py`
- **只训练 NAM** → `nam_train.py` (项目原始文件)

**Q: 如何选择合适的模型？**

- 想要可解释性：NAM, EBM, CART
- 想要高精度：XGBoost, DNN-MLP, EBM
- 想要快速训练：Logistic/Linear, CART
- 想要平衡：XGBoost, NAM

---

## 📚 更多文档

- 详细使用说明: `BASELINE_COMPARISON.md`
- NAM 论文: https://arxiv.org/abs/2004.13912
- 项目 README: `README.md`

---

**提示**: 第一次运行前，请确保安装所有依赖：

```bash
pip install xgboost scikit-learn interpret pandas numpy
```
