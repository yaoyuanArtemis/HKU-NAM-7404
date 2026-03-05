# 项目文件说明

## 🎯 主要文件（清晰命名）

| 文件 | 类型 | 作用 | 如何使用 |
|------|------|------|----------|
| **`run_experiment.py`** | ⭐ **主程序** | 单数据集对比实验 | `python run_experiment.py --data_path data.csv --target_column label` |
| **`run_all_datasets.py`** | ⭐ **批量运行** | 自动在所有NAM数据集上运行 | `python run_all_datasets.py` |
| **`baseline_models.py`** | 类库 | 提供模型训练类 | 被上面两个脚本导入使用 |

---

## 🚀 三种使用方式

### 方式 1: 单个数据集实验 ⭐ 推荐新手

```bash
# 使用自己的 CSV 数据
python run_experiment.py \
    --data_path your_data.csv \
    --target_column target \
    --task classification
```

**适合**:
- 快速测试单个数据集
- 使用自己的数据

---

### 方式 2: 批量运行所有 NAM 数据集 ⭐ 复现论文

```bash
# 自动在 8 个数据集上运行所有模型
python run_all_datasets.py
```

**自动完成**:
- ✅ 从 GCS 加载 8 个数据集（Telco, BreastCancer, Adult, Heart, Credit, Recidivism, Housing, Fico）
- ✅ 在每个数据集上训练 5-6 个模型
- ✅ 生成汇总报告和对比表格

**输出**:
```
all_results/
├── ALL_DATASETS_SUMMARY.csv    # 所有结果汇总
├── SUMMARY_REPORT.md            # Markdown 报告
├── Telco_temp_comparison.csv    # 各数据集详细结果
├── Heart_temp_comparison.csv
└── ...
```

**适合**:
- 复现 NAM 论文结果
- 全面对比所有模型和数据集

---

### 方式 3: 在 Colab 上运行

1. 上传项目到 Google Drive
2. 打开 `NAM_Baseline_Comparison.ipynb`
3. 运行单元格

详见: `COLAB_GUIDE.md`

---

## 📊 运行示例

### 示例 1: 快速测试单个数据集

```bash
# 使用 Breast Cancer 数据集
python -c "
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('breast_cancer.csv', index=False)
"

python run_experiment.py \
    --data_path breast_cancer.csv \
    --target_column target \
    --task classification
```

### 示例 2: 批量运行所有数据集

```bash
# 一键运行所有 NAM 论文数据集
python run_all_datasets.py

# 等待完成（约 10-30 分钟，取决于网络和计算资源）

# 查看汇总结果
cat all_results/SUMMARY_REPORT.md
```

### 示例 3: 只运行特定模型

```bash
# 只对比 XGBoost 和 CART
python run_experiment.py \
    --data_path data.csv \
    --target_column label \
    --models xgboost cart
```

---

## 🗂️ 完整项目结构

```
neural_additive_models/
│
├── 📊 主程序
│   ├── run_experiment.py         ⭐ 单数据集实验（主入口）
│   └── run_all_datasets.py       ⭐ 批量运行所有数据集
│
├── 📦 核心模块
│   ├── baseline_models.py        模型训练类库
│   ├── data_utils.py             NAM 数据集加载
│   ├── models.py                 NAM 模型定义
│   ├── graph_builder.py          NAM 计算图
│   └── nam_train.py              NAM 训练脚本
│
├── 📓 Jupyter Notebook
│   └── NAM_Baseline_Comparison.ipynb  Colab 使用
│
├── 📚 文档
│   ├── README_PROJECT.md         项目说明（本文件）
│   ├── COMPARISON_QUICK_START.md 快速开始
│   ├── COLAB_GUIDE.md            Colab 使用指南
│   ├── DATASETS.md               数据集详细说明
│   ├── FILES_OVERVIEW.md         文件总览
│   └── BASELINE_COMPARISON.md    完整文档
│
└── 📋 配置
    ├── requirements.txt          基础依赖
    └── requirements_baseline.txt 基线模型依赖
```

---

## 💡 常见场景

### 场景 1: 我想快速看看模型效果

```bash
python run_experiment.py --data_path my_data.csv --target_column label
```

### 场景 2: 我想复现 NAM 论文的对比实验

```bash
python run_all_datasets.py
```

### 场景 3: 我在 Colab 上运行

打开 `NAM_Baseline_Comparison.ipynb`

### 场景 4: 我只想对比几个模型

```bash
python run_experiment.py \
    --data_path data.csv \
    --target_column label \
    --models xgboost cart ebm
```

---

## 🔧 高级选项

### 调整数据分割比例

```bash
python run_experiment.py \
    --data_path data.csv \
    --target_column label \
    --test_size 0.15 \
    --val_size 0.15
```

### 指定输出目录

```bash
python run_experiment.py \
    --data_path data.csv \
    --target_column label \
    --output_dir ./my_results
```

### 修改随机种子

```bash
python run_experiment.py \
    --data_path data.csv \
    --target_column label \
    --random_state 123
```

---

## ❓ FAQ

### Q: 三个文件有什么区别？

| 文件 | 作用 |
|------|------|
| `baseline_models.py` | 类库，提供模型训练功能 |
| `run_experiment.py` | 单数据集实验主程序 |
| `run_all_datasets.py` | 批量运行多数据集 |

### Q: 我应该用哪个？

- **新手/单数据集** → `run_experiment.py`
- **复现论文/全面对比** → `run_all_datasets.py`
- **Colab** → `NAM_Baseline_Comparison.ipynb`

### Q: 数据集从哪来？

- `run_experiment.py`: 你提供 CSV 文件
- `run_all_datasets.py`: 自动从 GCS 下载 NAM 论文数据集

### Q: 批量运行要多久？

约 10-30 分钟，取决于：
- 网络速度（从 GCS 下载数据）
- 计算资源（8个数据集 × 5-6个模型）

---

## 📖 详细文档

- **快速开始**: `COMPARISON_QUICK_START.md`
- **数据集说明**: `DATASETS.md`
- **Colab 使用**: `COLAB_GUIDE.md`
- **完整文档**: `BASELINE_COMPARISON.md`
