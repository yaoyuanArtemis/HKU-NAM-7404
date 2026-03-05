# Google Colab 使用指南

## 📋 快速开始（3步）

### 1️⃣ 上传项目到 Google Drive

1. 打开 Google Drive: https://drive.google.com
2. 上传整个 `neural_additive_models` 文件夹
3. 记住文件夹路径，例如：
   ```
   我的云端硬盘/neural_additive_models
   ```

### 2️⃣ 在 Colab 中打开 Notebook

1. 在 Google Drive 中找到项目文件夹
2. 双击打开 `NAM_Baseline_Comparison.ipynb`
3. 选择 **"使用 Google Colaboratory 打开"**
4. 如果没有这个选项：
   - 右键点击文件 → 打开方式 → 连接更多应用
   - 搜索 "Colaboratory" → 安装

### 3️⃣ 运行实验

1. 点击菜单栏 **"代码执行程序" → "全部运行"**
2. 或者按顺序点击每个单元格左侧的播放按钮 ▶️
3. 第一次运行时会要求授权访问 Google Drive，点击允许

---

## 📁 项目结构

上传到 Google Drive 后的结构：

```
我的云端硬盘/
  └── neural_additive_models/         # 项目文件夹
      ├── NAM_Baseline_Comparison.ipynb  ⭐ 在 Colab 中打开这个文件
      ├── baseline_models.py
      ├── run_experiment.py
      ├── data_utils.py
      ├── models.py
      ├── nam_train.py
      ├── COMPARISON_QUICK_START.md
      ├── DATASETS.md
      └── requirements_baseline.txt
```

---

## 🎯 Notebook 功能

`NAM_Baseline_Comparison.ipynb` 包含：

### ✅ 自动配置
- 挂载 Google Drive
- 切换到项目目录
- 安装依赖包

### ✅ 三种运行方案
- **方案 A**: 使用示例数据（最快）
- **方案 B**: 上传你的 CSV 数据
- **方案 C**: 使用 sklearn 内置数据集

### ✅ 结果可视化
- 表格显示对比结果
- 性能对比柱状图
- 训练时间对比图

### ✅ 结果下载
- 自动打包结果文件
- 下载到本地

---

## 💡 使用示例

### 示例 1: 快速测试（最简单）

```python
# 在 Notebook 中运行这个单元格
!python baseline_models.py
```

### 示例 2: 使用自己的数据

```python
# 1. 上传 CSV 文件
from google.colab import files
uploaded = files.upload()

# 2. 运行对比
!python run_experiment.py \
    --data_path your_data.csv \
    --target_column target \
    --task classification
```

### 示例 3: 使用 Breast Cancer 数据集

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# 加载数据
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('breast_cancer.csv', index=False)

# 运行对比
!python run_experiment.py \
    --data_path breast_cancer.csv \
    --target_column target \
    --task classification
```

---

## 🔧 修改配置

### 修改项目路径

在 Notebook 的第 2 个单元格中：

```python
# 修改这里！
PROJECT_PATH = '/content/drive/MyDrive/你的文件夹名称/neural_additive_models'
```

### 修改目标列和任务类型

在运行对比时：

```python
TARGET_COLUMN = 'label'        # 改成你的目标列名
TASK = 'classification'        # 或 'regression'
```

### 只运行特定模型

```python
!python run_experiment.py \
    --data_path data.csv \
    --target_column target \
    --models xgboost cart mlp  # 只运行这三个模型
```

---

## 📊 查看结果

### 在 Notebook 中查看

```python
import pandas as pd

results = pd.read_csv('comparison_results/your_data_comparison.csv')
print(results)
```

### 下载到本地

```python
from google.colab import files

# 下载单个文件
files.download('comparison_results/your_data_comparison.csv')

# 或打包下载所有结果
!zip -r results.zip comparison_results/
files.download('results.zip')
```

---

## ⚠️ 常见问题

### Q1: 找不到项目文件？

**原因**: 项目路径设置错误

**解决**:
```python
# 检查当前路径
!pwd

# 列出 Drive 内容
!ls /content/drive/MyDrive/

# 修改 PROJECT_PATH 为正确路径
PROJECT_PATH = '/content/drive/MyDrive/正确的路径'
```

### Q2: 权限被拒绝？

**原因**: 未授权 Colab 访问 Google Drive

**解决**: 重新运行挂载单元格，点击授权链接

### Q3: 依赖包安装失败？

**原因**: 网络问题或包冲突

**解决**:
```python
# 单独安装
!pip install xgboost
!pip install scikit-learn
!pip install interpret
```

### Q4: 内存不足？

**原因**: 数据集太大或模型太多

**解决**:
```python
# 1. 只运行部分模型
!python run_experiment.py \
    --data_path data.csv \
    --target_column target \
    --models logistic cart  # 只跑小模型

# 2. 使用更小的数据集
df_small = df.sample(n=1000)  # 采样1000条
df_small.to_csv('data_small.csv', index=False)
```

### Q5: 如何使用 GPU？

Colab 默认使用 CPU，基线模型对比不需要 GPU。

如果需要训练 NAM（TensorFlow）：
1. 菜单栏：代码执行程序 → 更改运行时类型
2. 硬件加速器：选择 GPU
3. 保存

---

## 📝 完整工作流程

```python
# === 1. 环境设置 ===
from google.colab import drive, files
import os
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/neural_additive_models')

# === 2. 安装依赖 ===
!pip install -q xgboost scikit-learn interpret

# === 3. 准备数据 ===
# 选项 A: 上传文件
uploaded = files.upload()

# 选项 B: 使用 sklearn 数据
from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('data.csv', index=False)

# === 4. 运行对比 ===
!python run_experiment.py \
    --data_path data.csv \
    --target_column target \
    --task classification

# === 5. 查看结果 ===
results = pd.read_csv('comparison_results/data_comparison.csv')
print(results)

# === 6. 下载结果 ===
!zip -r results.zip comparison_results/
files.download('results.zip')
```

---

## 🚀 进阶技巧

### 使用 Python API

```python
from baseline_comparison import BaselineComparison
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 准备数据
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 运行对比
comparison = BaselineComparison(regression=False)
results = comparison.train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test
)

print(results)
```

### 结果可视化

```python
import matplotlib.pyplot as plt

# 性能对比
plt.figure(figsize=(10, 6))
plt.barh(results['Model'], results['Test AUROC'])
plt.xlabel('Test AUROC')
plt.title('Model Performance Comparison')
plt.show()

# 训练时间对比
plt.figure(figsize=(10, 6))
plt.barh(results['Model'], results['Train Time (s)'])
plt.xlabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.show()
```

---

## 📚 相关文档

- 快速开始指南: `COMPARISON_QUICK_START.md`
- 数据集说明: `DATASETS.md`
- 基线对比文档: `BASELINE_COMPARISON.md`
- 文件总览: `FILES_OVERVIEW.md`

---

## 💬 需要帮助？

1. 查看 Notebook 中的 "常见问题" 部分
2. 阅读项目文档
3. 检查是否正确设置了路径和参数

---

**提示**: 第一次使用 Colab 时，建议先运行方案 A（示例数据）熟悉流程，再使用自己的数据。
