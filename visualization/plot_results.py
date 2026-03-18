"""
Visualization functions for AutoML experiment results.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# Color map for optimizers
OPTIMIZER_COLORS = {
    'GA':  '#E63946',
    'GP':  '#F4A261',
    'DE':  '#2A9D8F',
    'PSO': '#457B9D',
    'ACO': '#6A0572',
    'ABC': '#F7B731',
    'GWO': '#264653',
    'WOA': '#2980B9',
    'HHO': '#8E44AD',
    'CS':  '#27AE60',
}

OPTIMIZER_STYLES = {
    'GA': '-', 'GP': '-', 'DE': '-', 'PSO': '-',
    'ACO': '-', 'ABC': '-', 'GWO': '-',
    'WOA': '--', 'HHO': '--', 'CS': '--'
}


def plot_convergence_curves(convergence_data, dataset_name, save_path, show=False):
    """
    Plot convergence curves for all optimizers on a dataset.

    Args:
        convergence_data: dict from runner.convergence_data
        dataset_name: string for title
        save_path: file path to save PNG
        show: whether to display the plot

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = False
    for key, data in convergence_data.items():
        if data['dataset'] != dataset_name:
            continue

        opt_name = data['optimizer']
        curves = data['curves']
        if not curves:
            continue

        # Average across runs
        max_len = max(len(c) for c in curves)
        padded = [c + [c[-1]] * (max_len - len(c)) if c else [0] * max_len for c in curves]
        mean_curve = np.mean(padded, axis=0)
        std_curve = np.std(padded, axis=0)

        color = OPTIMIZER_COLORS.get(opt_name, '#333333')
        style = OPTIMIZER_STYLES.get(opt_name, '-')

        x = np.arange(1, len(mean_curve) + 1)
        ax.plot(x, mean_curve, style, color=color, label=opt_name, linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=color)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, 'No convergence data available',
                ha='center', va='center', transform=ax.transAxes)

    ax.set_title(f'Convergence Curves — {dataset_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_comparison_bar(summary_df, metric, dataset_name, save_path, show=False):
    """
    Bar chart comparing optimizers on a metric for a dataset.

    Args:
        summary_df: DataFrame with results
        metric: column name to plot
        dataset_name: filter by dataset
        save_path: PNG output path

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df = summary_df[summary_df['dataset'] == dataset_name] if 'dataset' in summary_df.columns else summary_df

    if df.empty or metric not in df.columns:
        ax.text(0.5, 0.5, f'No data for metric: {metric}',
                ha='center', va='center', transform=ax.transAxes)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return fig

    # Aggregate by optimizer
    agg = df.groupby('optimizer')[metric].agg(['mean', 'std']).reset_index()
    opts = agg['optimizer'].tolist()
    means = agg['mean'].tolist()
    stds = agg['std'].fillna(0).tolist()

    colors = [OPTIMIZER_COLORS.get(o, '#888888') for o in opts]

    bars = ax.bar(opts, means, yerr=stds, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5, capsize=4)

    ax.set_title(f'{metric.replace("_", " ").title()} Comparison — {dataset_name.title()}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Optimizer', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_heatmap(summary_df, metric, save_path, show=False):
    """
    Heatmap: rows=optimizers, cols=datasets, values=metric.

    Args:
        summary_df: DataFrame with columns: optimizer, dataset, metric
        metric: column to visualize
        save_path: PNG output path

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if metric not in summary_df.columns:
        ax.text(0.5, 0.5, f'Metric {metric} not found',
                ha='center', va='center', transform=ax.transAxes)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return fig

    pivot = summary_df.groupby(['optimizer', 'dataset'])[metric].mean().unstack(fill_value=0)

    if pivot.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return fig

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, annot_kws={'size': 9})

    ax.set_title(f'Performance Heatmap — {metric.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Optimizer', fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_time_vs_performance(summary_df, save_path, task_filter=None, show=False):
    """
    Scatter plot: x=mean_time, y=mean_test_score, annotated by optimizer.

    Args:
        summary_df: DataFrame with results
        save_path: PNG output path
        task_filter: 'classification' or 'regression' or None for all

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df = summary_df.copy()
    if task_filter and 'task' in df.columns:
        df = df[df['task'] == task_filter]

    # Determine score metric
    score_col = None
    for col in ['test_accuracy', 'test_r2', 'cv_fitness']:
        if col in df.columns:
            score_col = col
            break

    if score_col is None or 'total_time' not in df.columns:
        ax.text(0.5, 0.5, 'Insufficient data for scatter plot',
                ha='center', va='center', transform=ax.transAxes)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return fig

    agg = df.groupby('optimizer').agg(
        mean_time=('total_time', 'mean'),
        mean_score=(score_col, 'mean')
    ).reset_index()

    for _, row in agg.iterrows():
        opt = row['optimizer']
        color = OPTIMIZER_COLORS.get(opt, '#888888')
        ax.scatter(row['mean_time'], row['mean_score'], color=color, s=150,
                   zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(opt, (row['mean_time'], row['mean_score']),
                    textcoords='offset points', xytext=(8, 4), fontsize=9)

    ax.set_title('Time vs Performance Scatter', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mean Time (s)', fontsize=12)
    ax.set_ylabel(score_col.replace('_', ' ').title(), fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_regression_metrics(summary_df, dataset_name, save_path, show=False):
    """
    3-panel subplot: RMSE, MAE, R² per optimizer for regression datasets.

    Args:
        summary_df: DataFrame with test_rmse, test_mae, test_r2 columns
        dataset_name: dataset to filter
        save_path: PNG output path

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    df = summary_df[summary_df['dataset'] == dataset_name] if 'dataset' in summary_df.columns else summary_df

    metrics_info = [
        ('test_rmse', 'RMSE', axes[0], 'lower is better'),
        ('test_mae', 'MAE', axes[1], 'lower is better'),
        ('test_r2', 'R²', axes[2], 'higher is better'),
    ]

    for metric, label, ax, note in metrics_info:
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{label} not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        agg = df.groupby('optimizer')[metric].agg(['mean', 'std']).reset_index()
        opts = agg['optimizer'].tolist()
        means = agg['mean'].tolist()
        stds = agg['std'].fillna(0).tolist()
        colors = [OPTIMIZER_COLORS.get(o, '#888888') for o in opts]

        ax.bar(opts, means, yerr=stds, color=colors, alpha=0.85,
               edgecolor='black', linewidth=0.5, capsize=3)
        ax.set_title(f'{label}\n({note})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Optimizer', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Regression Metrics — {dataset_name.replace("_", " ").title()}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return fig


def generate_all_plots(all_results, convergence_data, output_dir='./results'):
    """Generate and save all required plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Build summary DataFrame
    rows = []
    for dataset_name, dataset_results in all_results.items():
        for opt_summary in dataset_results:
            for run_metrics in opt_summary.get('test_metrics', [{}]):
                row = {
                    'dataset': dataset_name,
                    'optimizer': opt_summary['optimizer'],
                    'cv_fitness': opt_summary['mean_fitness'],
                    'total_time': opt_summary['mean_time'],
                    'n_evals': opt_summary['mean_evals'],
                }
                row.update(run_metrics)
                rows.append(row)

    if not rows:
        print("No results to plot.")
        return

    summary_df = pd.DataFrame(rows)

    # 1. Convergence curves per dataset
    datasets = summary_df['dataset'].unique()
    for ds in datasets:
        save_path = os.path.join(output_dir, f'convergence_{ds}.png')
        plot_convergence_curves(convergence_data, ds, save_path)
        print(f"  📉 Convergence plot saved: {save_path}")

    # 2. Comparison bar charts
    task_metric_map = {
        'heart': 'test_accuracy', 'student': 'test_accuracy',
        'housing': 'test_rmse', 'diabetes': 'test_rmse'
    }
    for ds in datasets:
        metric = task_metric_map.get(ds, 'cv_fitness')
        if metric in summary_df.columns:
            save_path = os.path.join(output_dir, f'comparison_bar_{ds}.png')
            plot_comparison_bar(summary_df, metric, ds, save_path)
            print(f"  [chart] Bar chart saved: {save_path}")

    # 3. Heatmap
    for metric in ['test_accuracy', 'cv_fitness']:
        if metric in summary_df.columns:
            save_path = os.path.join(output_dir, f'heatmap_{metric}.png')
            plot_heatmap(summary_df, metric, save_path)
            print(f"  [map]️  Heatmap saved: {save_path}")
            break

    # 4. Time vs Performance scatter
    save_path = os.path.join(output_dir, 'time_vs_performance.png')
    plot_time_vs_performance(summary_df, save_path)
    print(f"  [time]️  Time vs Performance saved: {save_path}")

    # 5. Regression metrics (3-panel) for regression datasets
    reg_datasets = [ds for ds in datasets if ds in ['housing', 'diabetes']]
    for ds in reg_datasets:
        if 'test_rmse' in summary_df.columns:
            save_path = os.path.join(output_dir, f'regression_metrics_{ds}.png')
            plot_regression_metrics(summary_df, ds, save_path)
            print(f"  [ruler] Regression metrics plot saved: {save_path}")

    return summary_df
