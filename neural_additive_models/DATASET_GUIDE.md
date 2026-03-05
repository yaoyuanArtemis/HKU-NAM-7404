# NAM 数据集完整指南

## 📊 NAM 论文使用的 9 个数据集

### ✅ 可自动下载（6个）

| # | 数据集 | 来源 | 样本数 | 特征数 | 任务 |
|---|--------|------|--------|--------|------|
| 1 | Breast Cancer | sklearn | 569 | 30 | 分类 |
| 2 | California Housing | sklearn | ~20K | 8 | 回归 |
| 3 | Adult Income | UCI | ~48K | 14 | 分类 |
| 4 | Heart Disease | UCI | 303 | 13 | 分类 |
| 5 | Telco Churn | IBM | ~7K | 19 | 分类 |
| 6 | COMPAS Recidivism | ProPublica | ~6K | 11 | 分类 |

### ⚠️ 需要手动下载（2个）

| # | 数据集 | 说明 | 下载链接 |
|---|--------|------|----------|
| 7 | Credit Card Fraud | 需要 Kaggle 账号（150MB） | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| 8 | FICO Score | 需要注册 | https://community.fico.com/s/explainable-machine-learning-challenge |

### ❌ 需要授权（1个）

| # | 数据集 | 说明 | 链接 |
|---|--------|------|------|
| 9 | MIMIC-II | 需要 PhysioNet 认证 + DUA | https://mimic.mit.edu/ |

---

## 🚀 快速开始（3步）

### 第 1 步：下载数据集

```bash
python download_datasets.py
```

**输出**：
- `datasets/breast_cancer.csv` ✅
- `datasets/california_housing.csv` ✅
- `datasets/adult.csv` ✅
- `datasets/heart_disease.csv` ✅
- `datasets/telco_churn.csv` ✅
- `datasets/compas_recidivism.csv` ✅

### 第 2 步：运行单个数据集测试

```bash
python run_experiment.py \
    --data_path datasets/breast_cancer.csv \
    --target_column target \
    --task classification
```

### 第 3 步：批量运行所有数据集

```bash
python run_all_datasets.py
```

---

## 📥 手动下载数据集（可选）

### Credit Card Fraud

1. 访问: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. 登录 Kaggle 账号
3. 点击 "Download" 下载 `creditcard.csv`
4. 将文件移动到 `datasets/creditcard.csv`

### FICO Score

1. 访问: https://community.fico.com/s/explainable-machine-learning-challenge
2. 注册账号
3. 下载数据集
4. 按照说明预处理后保存

### MIMIC-II

1. 访问: https://mimic.mit.edu/docs/gettingstarted/
2. 完成 CITI 培训
3. 签署数据使用协议 (DUA)
4. 获得批准后下载

---

## 🔧 在 Colab 中使用

### 打开 Notebook

上传 `NAM_Complete_Workflow.ipynb` 到 Google Drive，在 Colab 中打开。

### 按顺序运行

1. **步骤 1**: 克隆/更新代码
2. **步骤 2**: 安装依赖
3. **步骤 3**: 下载数据集 ⭐
4. **步骤 4**: 运行实验
5. **步骤 5-7**: 查看、可视化、下载结果

---

## 📊 数据集详细信息

### 1. Breast Cancer Wisconsin

**任务**: 乳腺癌良恶性分类

**特征**: 肿瘤细胞核的各种测量值（半径、纹理、周长、面积等）

**来源**: sklearn 内置，自动生成

**使用**:
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

---

### 2. California Housing

**任务**: 预测加州房价中位数（回归）

**特征**: 收入中位数、房龄、房间数、地理位置等

**来源**: sklearn 内置，自动生成

---

### 3. Adult Income (Census)

**任务**: 预测年收入是否超过 $50K

**特征**: 年龄、教育、职业、工作时长等

**来源**: UCI ML Repository

**URL**: https://archive.ics.uci.edu/ml/datasets/Adult

---

### 4. Heart Disease

**任务**: 心脏病诊断

**特征**: 年龄、血压、胆固醇、心率等

**来源**: UCI ML Repository

**URL**: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

---

### 5. Telco Customer Churn

**任务**: 预测客户是否会取消服务

**特征**: 服务类型、合同期限、月费用等

**来源**: IBM 公开数据

**URL**: https://github.com/IBM/telco-customer-churn-on-icp4d

---

### 6. COMPAS Recidivism

**任务**: 预测再犯罪风险

**特征**: 犯罪历史、年龄、种族等

**来源**: ProPublica

**URL**: https://github.com/propublica/compas-analysis

**注意**: 涉及算法公平性问题

---

### 7. Credit Card Fraud

**任务**: 信用卡欺诈检测

**特征**: 交易金额、时间、PCA 变换后的特征

**特点**: 高度不平衡（欺诈占 0.172%）

**大小**: ~150MB

**来源**: Kaggle

**需要**: Kaggle 账号

---

### 8. FICO Score

**任务**: 预测信用评分

**特征**: 信贷历史、负债比率等

**来源**: FICO 官网

**需要**: 注册

---

### 9. MIMIC-II

**任务**: ICU 患者死亡率预测

**特征**: 前 48 小时生理指标

**来源**: PhysioNet

**需要**: 认证 + DUA

---

## ⚠️ 常见问题

### Q: 为什么不能从 GCS 直接下载？

A: GCS bucket `gs://nam_datasets/data` 访问受限（403 错误），需要从公开来源下载。

### Q: 哪些数据集可以立即使用？

A: 运行 `download_datasets.py` 后，6 个数据集可立即使用。

### Q: 我需要下载所有数据集吗？

A: 不需要。你可以只用可自动下载的 6 个数据集进行实验。

### Q: 手动下载的数据集如何使用？

A: 下载后放到 `datasets/` 目录，文件名匹配 `run_all_datasets.py` 中的配置。

---

## 📚 相关文档

- 项目说明: `README_PROJECT.md`
- Colab 使用: `COLAB_GUIDE.md`
- 快速开始: `COMPARISON_QUICK_START.md`
