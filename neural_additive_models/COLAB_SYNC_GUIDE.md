# Colab 自动同步指南

## 🎯 目标

本地修改代码 → `git push` → Colab 自动更新，**无需重新上传文件夹**！

---

## ✅ 方案对比

| 方案 | 同步方式 | 优点 | 缺点 |
|------|----------|------|------|
| **GitHub 同步** ⭐ | git push → git pull | 版本控制、最方便 | 需要 push |
| Google Drive 同步 | 自动同步 | 完全自动 | 速度慢、不稳定 |
| 重新上传 | 手动上传 | 简单直接 | 每次都要传 |

**推荐：GitHub 同步**

---

## 🚀 快速设置（3步）

### 第 1 步：本地提交代码到 GitHub

```bash
cd /Users/sh01679ml/Desktop/workding-code/google-research/neural_additive_models

# 添加所有文件
git add .

# 提交
git commit -m "初始提交: NAM 基线对比项目"

# 推送到 GitHub
git push origin main
```

### 第 2 步：在 Colab 中打开同步版 Notebook

1. 上传 `NAM_GitHub_Sync.ipynb` 到 Google Drive
2. 在 Colab 中打开
3. 运行第一个单元格（自动从 GitHub 克隆项目）

### 第 3 步：以后每次更新

**本地修改代码后**：
```bash
git add .
git commit -m "更新代码"
git push origin main
```

**在 Colab 中**：
运行第2个单元格（`git pull`）即可！

---

## 📋 完整工作流程

```
┌─────────────────┐
│   本地电脑      │
│                 │
│ 1. 修改代码     │
│ 2. git add .    │
│ 3. git commit   │
│ 4. git push     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    GitHub       │
│  (自动更新)     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   Colab         │
│                 │
│ git pull        │
│ (运行1个单元格)│
└─────────────────┘
```

---

## 💻 详细操作步骤

### 第一次设置

#### 1. 确保代码在 GitHub 上

```bash
cd /Users/sh01679ml/Desktop/workding-code/google-research/neural_additive_models

# 检查远程仓库
git remote -v
# 应该显示: https://github.com/yaoyuanArtemis/HKU-NAM-7404.git

# 检查当前分支
git branch
# 确保在 main 分支

# 推送代码
git push origin main
```

#### 2. 上传 Notebook 到 Google Drive

1. 把 `NAM_GitHub_Sync.ipynb` 上传到 Google Drive 任意位置
2. 双击打开 → 选择 "Google Colaboratory"

#### 3. 在 Colab 中首次运行

运行第1个单元格，会自动从 GitHub 克隆项目：

```python
# 这个单元格会执行
!git clone https://github.com/yaoyuanArtemis/HKU-NAM-7404.git /content/neural_additive_models
```

---

### 日常使用流程

#### 在本地修改代码

```bash
# 1. 修改文件（用 VSCode/PyCharm 等）
# 例如：修改 run_experiment.py

# 2. 提交修改
cd /Users/sh01679ml/Desktop/workding-code/google-research/neural_additive_models
git add .
git commit -m "修改实验参数"
git push origin main
```

#### 在 Colab 中更新

打开 `NAM_GitHub_Sync.ipynb`，运行第2个单元格：

```python
# 这个单元格会执行
!git pull origin main
```

就这么简单！✨

---

## 📊 对比：旧方式 vs 新方式

### ❌ 旧方式（每次上传）

```
1. 本地修改代码
2. 压缩整个文件夹
3. 上传到 Google Drive
4. 在 Drive 中解压
5. 在 Colab 中切换路径
```

**耗时**: ~5-10 分钟每次

### ✅ 新方式（Git 同步）

```
1. 本地修改代码
2. git push (1条命令)
3. Colab 中 git pull (运行1个单元格)
```

**耗时**: ~10 秒！

---

## 🔧 常用 Git 命令速查

### 提交代码

```bash
git add .                    # 添加所有修改
git add run_experiment.py    # 添加单个文件
git commit -m "修改说明"     # 提交
git push origin main         # 推送到 GitHub
```

### 查看状态

```bash
git status                   # 查看修改状态
git log --oneline -5         # 查看最近5次提交
git diff                     # 查看具体修改
```

### 同步代码

```bash
git pull origin main         # 从 GitHub 拉取最新代码
```

### 放弃本地修改

```bash
git reset --hard origin/main # 完全恢复到远程版本
```

---

## 💡 高级技巧

### 1. 在 Colab 中自动更新并运行

把这个放在一个单元格中：

```python
import os
os.chdir('/content/neural_additive_models')

# 更新代码
!git pull origin main

# 运行实验
!python run_experiment.py --data_path data.csv --target_column label
```

一键搞定！

### 2. 批量实验脚本

```python
# 更新代码
!git pull origin main

# 批量运行所有数据集
!python run_all_datasets.py

# 下载结果
from google.colab import files
!zip -r results.zip all_results/
files.download('results.zip')
```

### 3. 定时自动更新（可选）

```python
import time

def auto_update_and_run():
    !git pull origin main
    # 运行实验
    !python run_experiment.py --data_path data.csv --target_column label

# 每小时更新一次
while True:
    auto_update_and_run()
    time.sleep(3600)
```

---

## ⚠️ 常见问题

### Q1: git pull 提示冲突

**原因**: Colab 中临时修改了文件

**解决**:
```python
# 方案1: 放弃 Colab 中的修改
!git reset --hard origin/main

# 方案2: 保存 Colab 修改后再拉取
!git stash        # 暂存修改
!git pull         # 拉取更新
!git stash pop    # 恢复修改
```

### Q2: 忘记 push，Colab 中没有最新代码

**解决**: 回到本地执行 `git push`，然后在 Colab 执行 `git pull`

### Q3: GitHub 私有仓库需要验证

**解决**: 在 Colab 中设置 Personal Access Token

```python
# 使用 token 克隆私有仓库
!git clone https://<your_token>@github.com/yaoyuanArtemis/HKU-NAM-7404.git
```

Token 获取方式：GitHub → Settings → Developer settings → Personal access tokens

### Q4: 想在 Colab 中临时测试，不影响 GitHub

**解决**: 直接在 Colab 中修改即可，这些修改不会提交到 GitHub（除非你主动 push）

### Q5: 大文件（数据集、模型）怎么办？

**方案**:
1. 大文件不要 commit 到 Git（用 `.gitignore`）
2. 使用 GCS/S3 存储数据
3. 或用 Git LFS（Large File Storage）

---

## 📝 .gitignore 配置

创建 `.gitignore` 文件，避免提交不必要的文件：

```bash
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# 数据文件
*.csv
*.h5
*.pkl
data/
datasets/

# 结果文件
comparison_results/
all_results/
*.zip

# Jupyter
.ipynb_checkpoints/

# 系统文件
.DS_Store
Thumbs.db
```

保存为 `.gitignore`，然后：

```bash
git add .gitignore
git commit -m "添加 .gitignore"
git push origin main
```

---

## 🎓 学习资源

- Git 基础教程: https://git-scm.com/book/zh/v2
- GitHub 指南: https://guides.github.com/
- Git 速查表: https://training.github.com/downloads/zh_CN/github-git-cheat-sheet/

---

## ✅ 检查清单

设置完成后，检查这些：

- [ ] 本地代码已 push 到 GitHub
- [ ] `NAM_GitHub_Sync.ipynb` 已上传到 Google Drive
- [ ] 在 Colab 中成功克隆项目
- [ ] 能够 `git pull` 更新代码
- [ ] 能够运行实验脚本

全部打勾后，你就可以无缝同步了！🎉
