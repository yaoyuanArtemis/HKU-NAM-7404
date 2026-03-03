# 模型对比文件总览

## 📝 Python 脚本（仅2个）

| 文件 | 用途 | 推荐度 | 说明 |
|------|------|--------|------|
| `compare_all_models.py` | 命令行对比工具 | ⭐⭐⭐ | 最推荐 |
| `baseline_comparison.py` | Python 核心类库 | ⭐⭐⭐ | 必需依赖 |

## 📚 文档（3个）

| 文件 | 内容 | 适合人群 |
|------|------|----------|
| `COMPARISON_QUICK_START.md` | 快速开始指南 | 新手必看 |
| `BASELINE_COMPARISON.md` | 完整使用文档 | 深入了解 |
| `requirements_baseline.txt` | 依赖包列表 | 安装依赖 |

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install xgboost scikit-learn interpret pandas numpy

# 2. 运行对比
python compare_all_models.py --data_path data.csv --target_column label

# 3. 查看结果
cat comparison_results/data_comparison.csv
```

## 💡 如何使用？

- **命令行对比** → `compare_all_models.py` （一键搞定，推荐）
- **代码集成** → `baseline_comparison.py` （Python API）

## 📖 详细文档

查看 `COMPARISON_QUICK_START.md` 了解更多使用方法。
