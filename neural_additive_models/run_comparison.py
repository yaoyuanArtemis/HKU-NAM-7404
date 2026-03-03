"""Example script to run baseline comparison with your own data.

This script demonstrates how to:
1. Load your dataset
2. Run baseline models (Logistic/Linear, CART, XGBoost, DNN-MLP, EBM)
3. Compare results
4. Save comparison results

Usage:
    python run_comparison.py --data_path <path> --task classification
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_comparison import BaselineComparison
from data_utils import load_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run baseline model comparison'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to dataset file (CSV)'
    )

    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'regression'],
        default='classification',
        help='Type of task: classification or regression'
    )

    parser.add_argument(
        '--target_column',
        type=str,
        default='target',
        help='Name of target column in dataset'
    )

    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of dataset to use for testing'
    )

    parser.add_argument(
        '--val_size',
        type=float,
        default=0.2,
        help='Proportion of training data to use for validation'
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./comparison_results',
        help='Directory to save comparison results'
    )

    return parser.parse_args()


def load_data(data_path: str, target_column: str):
    """Load dataset from CSV file.

    Args:
        data_path: Path to CSV file
        target_column: Name of target column

    Returns:
        X: Features (numpy array)
        y: Labels (numpy array)
        feature_names: List of feature names
    """
    df = pd.read_csv(data_path)

    # Separate features and target
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    feature_names = df.drop(columns=[target_column]).columns.tolist()

    return X, y, feature_names


def main():
    """Main function to run baseline comparison."""
    args = parse_args()

    print("="*70)
    print("BASELINE MODEL COMPARISON")
    print("="*70)
    print(f"Dataset: {args.data_path}")
    print(f"Task: {args.task}")
    print(f"Random seed: {args.random_state}")
    print("="*70)

    # Load data
    print("\nLoading data...")
    X, y, feature_names = load_data(args.data_path, args.target_column)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of samples: {len(y)}")

    if args.task == 'classification':
        unique_labels = np.unique(y)
        print(f"Number of classes: {len(unique_labels)}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    else:
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

    # Split data
    print("\nSplitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if args.task == 'classification' else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_temp if args.task == 'classification' else None
    )

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    # Initialize comparison
    print("\nInitializing baseline models...")
    regression = (args.task == 'regression')
    comparison = BaselineComparison(
        regression=regression,
        random_state=args.random_state
    )

    # Train and evaluate
    print("\nTraining and evaluating models...")
    results_df = comparison.train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)

    # Save results
    dataset_name = args.data_path.split('/')[-1].replace('.csv', '')
    comparison.save_results(args.output_dir, dataset_name)

    print("\nComparison complete!")


if __name__ == '__main__':
    main()
