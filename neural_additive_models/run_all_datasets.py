"""批量在多个数据集上运行基线模型对比实验。

使用下载到本地的数据集（而不是从 GCS 读取）。

Usage:
    # 首先下载数据集
    python download_datasets.py

    # 然后运行实验
    python run_all_datasets.py
"""

import os
import subprocess
import time
from datetime import datetime
import pandas as pd


# 数据目录
DATA_DIR = './datasets'


# NAM 论文使用的数据集配置（使用本地文件）
DATASETS = {
    # 分类任务（sklearn 内置，自动生成）
    'BreastCancer': {
        'file': 'breast_cancer.csv',
        'task': 'classification',
        'target': 'target'
    },

    # 分类任务（可从公开源下载）
    'Adult': {
        'file': 'adult.csv',
        'task': 'classification',
        'target': 'income'
    },
    'Heart': {
        'file': 'heart_disease.csv',
        'task': 'classification',
        'target': 'target'
    },
    'Telco': {
        'file': 'telco_churn.csv',
        'task': 'classification',
        'target': 'Churn'
    },
    'Recidivism': {
        'file': 'compas_recidivism.csv',
        'task': 'classification',
        'target': 'two_year_recid'
    },

    # 分类任务（需要手动下载）
    'Credit': {
        'file': 'creditcard.csv',
        'task': 'classification',
        'target': 'Class',
        'manual': True  # 需要手动下载
    },

    # 回归任务
    'Housing': {
        'file': 'california_housing.csv',
        'task': 'regression',
        'target': 'target'
    },
}


def run_experiment_on_dataset(dataset_name, config, output_dir='./all_results'):
    """在单个数据集上运行实验。

    Args:
        dataset_name: 数据集名称
        config: 数据集配置 (file, task, target)
        output_dir: 结果输出目录

    Returns:
        是否成功运行
    """
    print("\n" + "="*70)
    print(f"📊 运行数据集: {dataset_name}")
    print("="*70)

    # 检查是否需要手动下载
    if config.get('manual'):
        print(f"⚠️  {dataset_name} 需要手动下载")
        print(f"   请先运行: python download_datasets.py")
        print(f"   并按照提示手动下载")
        return False

    # 检查数据文件是否存在
    data_file = os.path.join(DATA_DIR, config['file'])

    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print(f"   请先运行: python download_datasets.py")
        return False

    print(f"✓ 找到数据文件: {data_file}")

    # 读取数据查看基本信息
    try:
        df = pd.read_csv(data_file)
        print(f"  数据形状: {df.shape}")
        print(f"  目标列: {config['target']}")

        if config['target'] not in df.columns:
            print(f"  ⚠️  目标列 '{config['target']}' 不存在")
            print(f"  可用列: {list(df.columns)}")
            return False

    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return False

    # 运行对比实验
    cmd = [
        'python', 'run_experiment.py',
        '--data_path', data_file,
        '--target_column', config['target'],
        '--task', config['task'],
        '--output_dir', output_dir
    ]

    print(f"\n运行命令: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ {dataset_name} 完成！用时: {elapsed_time:.1f}秒")
            return True
        else:
            print(f"❌ {dataset_name} 失败！")
            print(f"错误信息:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset_name} 超时（10分钟）")
        return False
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False


def collect_all_results(output_dir='./all_results'):
    """收集所有数据集的结果并生成汇总报告。

    Args:
        output_dir: 结果目录

    Returns:
        汇总的 DataFrame
    """
    import glob

    print("\n" + "="*70)
    print("📊 收集所有结果...")
    print("="*70)

    result_files = glob.glob(f'{output_dir}/*_comparison.csv')

    if not result_files:
        print("未找到结果文件")
        return None

    all_results = []

    for result_file in sorted(result_files):
        dataset_name = os.path.basename(result_file).replace('_temp_comparison.csv', '')

        try:
            df = pd.read_csv(result_file)
            df['Dataset'] = dataset_name
            all_results.append(df)
            print(f"✓ {dataset_name}")
        except Exception as e:
            print(f"❌ 读取 {dataset_name} 失败: {e}")

    if not all_results:
        return None

    # 合并所有结果
    summary_df = pd.concat(all_results, ignore_index=True)

    # 重新排列列顺序
    cols = ['Dataset', 'Model'] + [col for col in summary_df.columns if col not in ['Dataset', 'Model']]
    summary_df = summary_df[cols]

    return summary_df


def generate_summary_report(summary_df, output_dir='./all_results'):
    """生成汇总报告。

    Args:
        summary_df: 汇总的结果 DataFrame
        output_dir: 输出目录
    """
    if summary_df is None or summary_df.empty:
        print("没有结果可以汇总")
        return

    print("\n" + "="*70)
    print("📄 生成汇总报告...")
    print("="*70)

    # 保存完整结果
    summary_path = os.path.join(output_dir, 'ALL_DATASETS_SUMMARY.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ 完整结果: {summary_path}")

    # 生成 Markdown 报告
    report_path = os.path.join(output_dir, 'SUMMARY_REPORT.md')

    with open(report_path, 'w') as f:
        f.write("# 所有数据集模型对比汇总报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 按数据集分组
        for dataset in summary_df['Dataset'].unique():
            f.write(f"## {dataset}\n\n")

            dataset_df = summary_df[summary_df['Dataset'] == dataset]
            dataset_df = dataset_df.drop(columns=['Dataset'])

            f.write(dataset_df.to_markdown(index=False))
            f.write("\n\n")

        # 总体统计
        f.write("## 总体统计\n\n")
        f.write(f"- 数据集数量: {summary_df['Dataset'].nunique()}\n")
        f.write(f"- 模型数量: {summary_df['Model'].nunique()}\n")
        f.write(f"- 总实验次数: {len(summary_df)}\n\n")

        # 最佳模型统计
        metric_cols = [col for col in summary_df.columns if 'Test' in col and ('AUROC' in col or 'RMSE' in col)]

        if metric_cols:
            metric_col = metric_cols[0]
            ascending = 'RMSE' in metric_col

            f.write(f"### 各数据集最佳模型 (按 {metric_col})\n\n")
            f.write("| Dataset | Best Model | Score |\n")
            f.write("|---------|------------|-------|\n")

            for dataset in summary_df['Dataset'].unique():
                dataset_df = summary_df[summary_df['Dataset'] == dataset]

                if ascending:
                    best_idx = dataset_df[metric_col].idxmin()
                else:
                    best_idx = dataset_df[metric_col].idxmax()

                best_model = dataset_df.loc[best_idx, 'Model']
                best_score = dataset_df.loc[best_idx, metric_col]

                f.write(f"| {dataset} | {best_model} | {best_score:.4f} |\n")

    print(f"✓ Markdown 报告: {report_path}")

    # 打印简要结果
    print("\n" + "="*70)
    print("📊 简要结果")
    print("="*70)
    print(f"运行的数据集: {summary_df['Dataset'].nunique()}")
    print(f"对比的模型: {summary_df['Model'].nunique()}")
    print(f"\n数据集列表: {', '.join(sorted(summary_df['Dataset'].unique()))}")
    print(f"模型列表: {', '.join(sorted(summary_df['Model'].unique()))}")


def main():
    """主函数：批量运行所有数据集。"""
    print("="*70)
    print("🚀 批量运行 NAM 基线模型对比实验")
    print("="*70)
    print(f"数据集数量: {len(DATASETS)}")
    print(f"数据集列表: {', '.join(DATASETS.keys())}")
    print(f"数据目录: {DATA_DIR}")
    print("")

    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        print(f"\n请先运行: python download_datasets.py")
        return

    # 创建输出目录
    output_dir = './all_results'
    os.makedirs(output_dir, exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    # 运行统计
    success_count = 0
    failed_datasets = []

    # 遍历所有数据集
    for dataset_name, config in DATASETS.items():
        success = run_experiment_on_dataset(dataset_name, config, output_dir)

        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)

        # 短暂暂停，避免过载
        time.sleep(1)

    # 总用时
    total_time = time.time() - start_time

    # 收集和汇总结果
    summary_df = collect_all_results(output_dir)

    if summary_df is not None:
        generate_summary_report(summary_df, output_dir)

    # 打印最终统计
    print("\n" + "="*70)
    print("✅ 批量实验完成！")
    print("="*70)
    print(f"总用时: {total_time/60:.1f} 分钟")
    print(f"成功: {success_count}/{len(DATASETS)}")

    if failed_datasets:
        print(f"失败的数据集: {', '.join(failed_datasets)}")

    print(f"\n结果保存在: {output_dir}")
    print("  - ALL_DATASETS_SUMMARY.csv  (完整结果)")
    print("  - SUMMARY_REPORT.md         (Markdown 报告)")
    print("")


if __name__ == '__main__':
    main()
