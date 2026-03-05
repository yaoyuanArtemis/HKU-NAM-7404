# Colab 同步快速开始（3步）

## 📍 你的 GitHub 配置

- **仓库**: https://github.com/yaoyuanArtemis/HKU-NAM-7404
- **分支**: master
- **项目路径**: neural_additive_models/

---

## 🚀 快速设置（3步）

### 第 1 步：提交代码到 GitHub

```bash
cd /Users/sh01679ml/Desktop/workding-code/google-research/neural_additive_models

# 添加所有文件
git add .

# 提交
git commit -m "添加基线对比功能"

# 推送（注意：是 master 分支）
git push origin master
```

### 第 2 步：上传 Notebook 到 Google Drive

1. 把 `NAM_GitHub_Sync.ipynb` 上传到 Google Drive 任意位置
2. 双击打开 → 选择 "Google Colaboratory"

### 第 3 步：在 Colab 中运行

打开 `NAM_GitHub_Sync.ipynb`，**按顺序运行**：

1. **单元格 1**: 克隆项目（首次运行）
2. **单元格 2**: 更新代码 ⭐ **每次运行前必须执行**
3. **单元格 3**: 安装依赖
4. **单元格 4**: 运行实验

完成！✨

---

## 🔄 日常使用流程

### 本地修改代码

```bash
cd /Users/sh01679ml/Desktop/workding-code/google-research/neural_additive_models

# 修改代码...

# 提交
git add .
git commit -m "修改实验参数"
git push origin master  # 注意：是 master 分支
```

### Colab 中同步

打开 `NAM_GitHub_Sync.ipynb`，运行第2个单元格（更新代码）：

```python
!git pull origin master
```

就这么简单！

---

## 📊 验证同步

### 检查本地是否已推送

```bash
git status
# 应该显示：nothing to commit, working tree clean

git log -1
# 查看最新提交
```

### 在 Colab 中验证

```python
!git log -3 --oneline
# 应该看到你刚才的提交
```

---

## ⚡ 一键脚本

### 本地：提交并推送

创建 `push.sh`：

```bash
#!/bin/bash
cd /Users/sh01679ml/Desktop/workding-code/google-research/neural_additive_models
git add .
git commit -m "更新: $(date '+%Y-%m-%d %H:%M')"
git push origin master
echo "✓ 已推送到 GitHub"
```

使用：
```bash
chmod +x push.sh
./push.sh
```

### Colab：更新并运行

在 Notebook 中有一个 "🚀 一键更新并运行" 单元格，运行即可！

---

## 🎯 完整工作流示例

### 场景：我修改了 run_experiment.py，想在 Colab 上测试

```bash
# 1. 本地修改代码
vim run_experiment.py

# 2. 提交
git add run_experiment.py
git commit -m "优化实验参数"
git push origin master

# 3. 打开 Colab，运行 "🚀 一键更新并运行" 单元格
# ✓ 完成！代码自动更新并运行
```

---

## 💡 Pro Tips

### 1. 使用 .gitignore

我已经创建了 `.gitignore`，避免提交：
- 数据文件 (*.csv)
- 结果文件 (comparison_results/)
- Python 缓存 (__pycache__/)

### 2. 查看文件是否会被提交

```bash
git status
```

### 3. 撤销本地修改

```bash
# 撤销所有未提交的修改
git reset --hard

# 撤销单个文件
git checkout -- run_experiment.py
```

### 4. 查看修改内容

```bash
# 查看修改了什么
git diff

# 查看某个文件的修改
git diff run_experiment.py
```

---

## ⚠️ 注意事项

1. **分支名是 master**（不是 main）
2. **项目在子目录** neural_additive_models/ 下
3. **首次克隆** 会下载整个仓库，包含其他目录
4. **大文件不要提交**（已在 .gitignore 中配置）

---

## 🆘 常见问题快速修复

### Git push 被拒绝

```bash
git pull origin master  # 先拉取
git push origin master  # 再推送
```

### Colab 中 git pull 冲突

在 Notebook 中运行：
```python
!git reset --hard origin/master
```

### 忘记推送

```bash
git push origin master
```

然后在 Colab 中 `git pull`

---

## ✅ 检查清单

设置完成后检查：

- [ ] 本地代码已推送：`git status` 显示 clean
- [ ] GitHub 上能看到最新提交
- [ ] `NAM_GitHub_Sync.ipynb` 已上传到 Google Drive
- [ ] 在 Colab 中能成功克隆项目
- [ ] 能够 `git pull` 更新代码
- [ ] 能够运行实验脚本

全部打勾，你就可以愉快地同步了！🎉

---

## 📖 相关文档

- 详细同步指南: `COLAB_SYNC_GUIDE.md`
- 项目使用说明: `README_PROJECT.md`
- Colab 使用指南: `COLAB_GUIDE.md`
