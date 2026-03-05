"""批量在多个数据集上运行基线模型对比实验。

Usage:
    python run_all_datasets.py
"""

import os
import subprocess
import time
from datetime import datetime
import pandas as pd


# NAM 论文使用的数据集配置
DATASETS = {
    # 分类任务
    'Telco': {'task': 'classification', 'target': 'Churn'},
    'BreastCancer': {'task': 'classification', 'target': 'target'},
    'Adult': {'task': 'classification', 'target': 'Income'},
    'Heart': {'task': 'classification', 'target': 'target'},
    'Credit': {'task': 'classification', 'target': 'Class'},
    'Recidivism': {'task': 'classification', 'target': 'is_recid'},

    # 回归任务
    'Housing': {'task': 'regression', 'target': 'target'},
    'Fico': {'task': 'regression', 'target': 'RiskPerformance'},
}

# MIMIC-II 需要特殊授权，默认跳过
SKIP_DATASETS = ['Mimic2']


def run_experiment_on_dataset(dataset_name, config, output_dir='./all_results'):
    """在单个数据集上运行实验。

    Args:
        dataset_name: 数据集名称
        config: 数据集配置 (task, target)
        output_dir: 结果输出目录

    Returns:
        是否成功运行
    """
    print("\n" + "="*70)
    print(f"📊 运行数据集: {dataset_name}")
    print("="*70)

    # 创建 CSV 文件（使用 data_utils 加载）
    try:
        from data_utils import load_dataset
        import numpy as np

        print(f"从 GCS 加载数据集...")
        dataset = load_dataset(dataset_name)
        X = dataset['X']
        y = dataset['y']

        # 保存为临时 CSV
        import pandas as pd
        df = X.copy()

        # 处理目标列
        if isinstance(y, pd.Series):
            df['target'] = y.values
        else:
            df['target'] = y

        temp_csv = f'/tmp/{dataset_name}_temp.csv'
        df.to_csv(temp_csv, index=False)

        print(f"✓ 数据集已加载: {X.shape[0]} 样本, {X.shape[1]} 特征")

    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return False

    # 运行对比实验
    cmd = [
        'python', 'run_experiment.py',
        '--data_path', temp_csv,
        '--target_column', 'target',
        '--task', config['task'],
        '--output_dir', output_dir
    ]

    print(f"运行命令: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ {dataset_name} 完成！用时: {elapsed_time:.1f}秒")

            # 清理临时文件
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

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
    print("")

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
        if dataset_name in SKIP_DATASETS:
            print(f"\n⏭️  跳过 {dataset_name} (需要特殊授权)")
            continue

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
    print(f"成功: {success_count}/{len(DATASETS) - len(SKIP_DATASETS)}")

    if failed_datasets:
        print(f"失败的数据集: {', '.join(failed_datasets)}")

    print(f"\n结果保存在: {output_dir}")
    print("  - ALL_DATASETS_SUMMARY.csv  (完整结果)")
    print("  - SUMMARY_REPORT.md         (Markdown 报告)")
    print("")


if __name__ == '__main__':
    main()
