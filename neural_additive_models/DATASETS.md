# NAM 论文数据集说明

## 📊 论文使用的数据集（9个）

根据 `data_utils.py` 中的代码，NAM 论文使用了以下 9 个数据集：

### 分类任务（7个）

| 数据集 | 名称 | 描述 | 样本数 | 特征数 |
|--------|------|------|--------|--------|
| **Telco** | Telco Customer Churn | 预测客户流失 | ~7K | 19 |
| **BreastCancer** | Breast Cancer Wisconsin | 乳腺癌诊断 | 569 | 30 |
| **Adult** | Adult Income (Census) | 收入预测 (>50K) | ~48K | 14 |
| **Credit** | Credit Fraud Detection | 信用卡欺诈检测 | ~284K | 30 |
| **Heart** | Heart Disease | 心脏病预测 | 303 | 13 |
| **Mimic2** | MIMIC-II ICU Mortality | ICU死亡率预测 | ~24K | 42 |
| **Recidivism** | ProPublica COMPAS | 再犯罪风险预测 | ~6K | 11 |
| **Fico** | FICO Score | 信用评分 | ~10K | 23 |

### 回归任务（1个）

| 数据集 | 名称 | 描述 | 样本数 | 特征数 |
|--------|------|------|--------|--------|
| **Housing** | California Housing | 加州房价预测 | ~20K | 8 |

---

## 📥 数据集获取方式

### ❌ 当前状态：数据集未下载

本地项目中**没有包含数据集文件**，需要从 Google Cloud Storage 下载。

### ✅ 下载数据集

数据集存储在公共 GCS bucket 中：`gs://nam_datasets/data`

#### 方法 1: 使用 gsutil（推荐）

```bash
# 1. 安装 gsutil
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash

# 2. 下载所有数据集到本地
mkdir -p ./data
gsutil -m cp -r gs://nam_datasets/data/* ./data/

# 3. 查看下载的文件
ls -lh ./data/
```

#### 方法 2: 从 Google Cloud Console 下载

访问：https://console.cloud.google.com/storage/browser/nam_datasets/data

手动下载需要的数据集文件。

#### 方法 3: 使用 wget/curl（如果数据集公开）

某些数据集可以从原始来源下载：

```bash
# Adult dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

# Breast Cancer (通过 sklearn)
# 这个数据集可以直接通过 sklearn 加载，不需要下载

# Heart Disease
wget https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
```

---

## 🔒 MIMIC-II 数据集特殊说明

**MIMIC-II** 数据集需要特殊授权：

1. 访问 PhysioNet 网站注册：https://mimic.mit.edu/docs/gettingstarted/
2. 完成 CITI 培训课程
3. 签署数据使用协议（DUA）
4. 获得批准后才能访问数据

如果需要 MIMIC-II 数据集，需要：
- 完成上述认证流程
- 联系 NAM 作者，提供你的 DUA 证明
- 获取预处理后的数据集

---

## 🛠️ 修改代码以使用本地数据

下载数据后，需要修改 `data_utils.py` 中的 `DATA_PATH`：

```python
# data_utils.py 第 39 行
# 原来：
DATA_PATH = 'gs://nam_datasets/data'

# 修改为：
DATA_PATH = './data'  # 或你的数据存储路径
```

或者在运行时指定数据路径（如果代码支持）。

---

## 📝 使用示例

### 使用 NAM 训练（假设已下载数据）

```bash
# 修改 data_utils.py 中的 DATA_PATH 后
python nam_train.py \
    --dataset_name Telco \
    --regression False \
    --training_epochs 1000 \
    --logdir ./logs/telco
```

### 可用的数据集名称

在代码中使用以下名称：

- `'Telco'`
- `'BreastCancer'`
- `'Adult'`
- `'Credit'`
- `'Heart'`
- `'Mimic2'` (需要授权)
- `'Recidivism'`
- `'Fico'`
- `'Housing'` (回归任务)

---

## 🎯 快速开始：使用自己的数据

如果你不想下载论文数据集，可以使用自己的数据：

### 1. 准备 CSV 格式数据

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
...
```

### 2. 使用对比脚本

```bash
python run_experiment.py \
    --data_path your_data.csv \
    --target_column target \
    --task classification
```

这样可以直接对比基线模型，无需下载论文数据集。

---

## 📚 数据集详细信息

### Adult (Census Income)
- **来源**: UCI ML Repository
- **任务**: 二分类（收入 >50K 或 ≤50K）
- **URL**: https://archive.ics.uci.edu/ml/datasets/Adult

### Telco Customer Churn
- **来源**: Kaggle
- **任务**: 客户流失预测
- **URL**: https://www.kaggle.com/blastchar/telco-customer-churn

### Credit Fraud Detection
- **来源**: Kaggle
- **任务**: 信用卡欺诈检测（高度不平衡）
- **URL**: https://www.kaggle.com/mlg-ulb/creditcardfraud

### Heart Disease
- **来源**: UCI ML Repository
- **任务**: 心脏病诊断
- **URL**: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

### COMPAS Recidivism
- **来源**: ProPublica
- **任务**: 再犯罪风险预测
- **URL**: https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis

### California Housing
- **来源**: scikit-learn built-in
- **任务**: 房价回归预测
- **代码**: `from sklearn.datasets import fetch_california_housing`

### Breast Cancer Wisconsin
- **来源**: scikit-learn built-in
- **任务**: 乳腺癌诊断
- **代码**: `from sklearn.datasets import load_breast_cancer`

---

## ⚠️ 注意事项

1. **数据集大小**: 某些数据集（如 Credit）较大，下载和处理可能需要时间
2. **预处理**: NAM 使用的是预处理后的数据集，直接从原始来源下载可能需要额外处理
3. **推荐方式**: 使用 `gsutil` 从 `gs://nam_datasets/data` 下载，确保数据格式一致
4. **MIMIC-II**: 如果不需要医疗数据集，可以跳过 MIMIC-II

---

## 🚀 最简单的方案

如果你想快速开始实验：

### 方案 1: 使用 sklearn 内置数据集

```python
from sklearn.datasets import load_breast_cancer, fetch_california_housing

# 分类任务
data = load_breast_cancer()
X, y = data.data, data.target

# 回归任务
data = fetch_california_housing()
X, y = data.data, data.target
```

### 方案 2: 使用自己的 CSV 数据

```bash
python run_experiment.py \
    --data_path my_data.csv \
    --target_column label
```

### 方案 3: 下载部分小数据集测试

```bash
# 只下载小数据集
gsutil cp gs://nam_datasets/data/HeartDisease.csv ./data/
gsutil cp gs://nam_datasets/data/WA_Fn-UseC_-Telco-Customer-Churn.csv ./data/
```

---

## 📞 需要帮助？

如果遇到数据集下载问题：

1. 查看 NAM 论文：https://arxiv.org/abs/2004.13912
2. 访问 GitHub Issues：https://github.com/google-research/google-research/issues
3. 使用论文作者提供的数据下载指南
