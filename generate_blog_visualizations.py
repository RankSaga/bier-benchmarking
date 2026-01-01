"""
Generate comprehensive visualizations for BEIR benchmarking blog post.
Creates publication-ready charts showing baseline vs optimized performance.
"""
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.patches as mpatches

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set style for professional charts
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# RankSaga color scheme (professional blue/purple tones)
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#7c3aed',    # Purple
    'accent': '#10b981',       # Green
    'warning': '#f59e0b',      # Orange
    'baseline': '#64748b',     # Gray
    'optimized': '#2563eb',    # Blue
    'improvement': '#10b981',  # Green
    'regression': '#ef4444',   # Red
}

# Create output directory
OUTPUT_DIR = Path(__file__).parent.parent / "public" / "images" / "blog" / "beir-benchmarking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths to result files
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"


def load_json_results(file_path: Path) -> Dict[str, Any]:
    """Load JSON results file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_all_results() -> Dict[str, Dict[str, Any]]:
    """Load all result files."""
    results = {}
    
    # Baseline
    baseline_file = RESULTS_DIR / "baseline" / "e5_base_v2_results.json"
    if baseline_file.exists():
        results['baseline'] = load_json_results(baseline_file)
    
    # Optimized
    optimized_file = RESULTS_DIR / "optimized" / "ranksaga_optimized_e5_v2_results.json"
    if optimized_file.exists():
        results['optimized'] = load_json_results(optimized_file)
    
    # Domain-specific
    scientific_file = RESULTS_DIR / "domain_specific" / "scientific" / "ranksaga_optimized_e5_v2_scientific_results.json"
    if scientific_file.exists():
        results['scientific'] = load_json_results(scientific_file)
    
    general_file = RESULTS_DIR / "domain_specific" / "general" / "ranksaga_optimized_e5_v2_general_results.json"
    if general_file.exists():
        results['general'] = load_json_results(general_file)
    
    return results


def create_comparison_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison DataFrame from results."""
    if 'baseline' not in results or 'optimized' not in results:
        logger.warning("Missing baseline or optimized results")
        return pd.DataFrame()
    
    baseline_data = results['baseline'].get('results', {})
    optimized_data = results['optimized'].get('results', {})
    
    comparison_data = []
    datasets = ['scifact', 'nfcorpus', 'scidocs', 'quora']
    metrics = ['NDCG@10', 'NDCG@100', 'MAP@100', 'Recall@100']
    
    for dataset in datasets:
        if dataset not in baseline_data or dataset not in optimized_data:
            continue
        
        baseline_metrics = baseline_data[dataset]
        optimized_metrics = optimized_data[dataset]
        
        for metric in metrics:
            baseline_val = baseline_metrics.get(metric, 0)
            optimized_val = optimized_metrics.get(metric, 0)
            
            if baseline_val > 0:
                improvement_pct = ((optimized_val - baseline_val) / baseline_val) * 100
            else:
                improvement_pct = 0.0
            
            comparison_data.append({
                'Dataset': dataset,
                'Metric': metric,
                'Baseline': baseline_val,
                'Optimized': optimized_val,
                'Improvement_%': improvement_pct,
                'Absolute_Improvement': optimized_val - baseline_val
            })
    
    return pd.DataFrame(comparison_data)


def plot_main_comparison(df: pd.DataFrame):
    """Create main performance comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RankSaga Model Optimization: Baseline vs Optimized Performance', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    datasets = df['Dataset'].unique()
    metrics = ['NDCG@10', 'NDCG@100', 'MAP@100', 'Recall@100']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        metric_df = df[df['Metric'] == metric].copy()
        
        x = np.arange(len(datasets))
        width = 0.35
        
        baseline_vals = [metric_df[metric_df['Dataset'] == d]['Baseline'].values[0] 
                        for d in datasets]
        optimized_vals = [metric_df[metric_df['Dataset'] == d]['Optimized'].values[0] 
                         for d in datasets]
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
                      color=COLORS['baseline'], alpha=0.8)
        bars2 = ax.bar(x + width/2, optimized_vals, width, label='RankSaga Optimized', 
                      color=COLORS['optimized'], alpha=0.8)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'main_performance_comparison.png', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved main performance comparison chart")
    plt.close()


def plot_improvement_heatmap(df: pd.DataFrame):
    """Create improvement percentage heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Dataset', columns='Metric', values='Improvement_%')
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement (%)'}, ax=ax, 
                linewidths=0.5, linecolor='gray', vmin=-30, vmax=55)
    
    ax.set_title('Performance Improvement Heatmap Across All Metrics', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Dataset', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'improvement_heatmap.png', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved improvement heatmap")
    plt.close()


def plot_improvement_percentages(df: pd.DataFrame):
    """Create horizontal bar chart showing improvement percentages."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Focus on NDCG@10 for main comparison
    ndcg10_df = df[df['Metric'] == 'NDCG@10'].copy()
    ndcg10_df = ndcg10_df.sort_values('Improvement_%', ascending=True)
    
    colors = [COLORS['improvement'] if x > 0 else COLORS['regression'] 
             for x in ndcg10_df['Improvement_%']]
    
    bars = ax.barh(ndcg10_df['Dataset'], ndcg10_df['Improvement_%'], color=colors, alpha=0.8)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Improvement (%)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('RankSaga Optimization: NDCG@10 Improvement by Dataset', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(ndcg10_df.iterrows()):
        val = row['Improvement_%']
        ax.text(val + (1 if val > 0 else -1), i, f'{val:.2f}%',
               va='center', ha='left' if val > 0 else 'right', fontsize=10, fontweight='bold')
    
    # Add legend
    improvement_patch = mpatches.Patch(color=COLORS['improvement'], label='Improvement')
    regression_patch = mpatches.Patch(color=COLORS['regression'], label='Regression')
    ax.legend(handles=[improvement_patch, regression_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'improvement_percentages.png', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved improvement percentages chart")
    plt.close()


def plot_scatter_comparison(df: pd.DataFrame):
    """Create scatter plot comparing baseline vs optimized."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline vs Optimized Performance Scatter Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['NDCG@10', 'NDCG@100', 'MAP@100', 'Recall@100']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        metric_df = df[df['Metric'] == metric].copy()
        
        # Create scatter plot
        scatter = ax.scatter(metric_df['Baseline'], metric_df['Optimized'], 
                           s=150, alpha=0.7, c=COLORS['optimized'], edgecolors='black', linewidth=1.5)
        
        # Add diagonal line (no improvement)
        max_val = max(metric_df['Baseline'].max(), metric_df['Optimized'].max())
        min_val = min(metric_df['Baseline'].min(), metric_df['Optimized'].min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, 
               label='No improvement')
        
        # Add dataset labels
        for _, row in metric_df.iterrows():
            ax.annotate(row['Dataset'], (row['Baseline'], row['Optimized']),
                       fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel(f'Baseline {metric}', fontsize=11)
        ax.set_ylabel(f'Optimized {metric}', fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_comparison.png', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved scatter comparison chart")
    plt.close()


def plot_radar_charts(df: pd.DataFrame):
    """Create radar charts for each dataset."""
    from math import pi
    
    datasets = df['Dataset'].unique()
    metrics = ['NDCG@10', 'NDCG@100', 'MAP@100', 'Recall@100']
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each metric
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    fig.suptitle('Multi-Metric Performance Comparison: Radar Charts', 
                 fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[df['Dataset'] == dataset]
        
        baseline_vals = []
        optimized_vals = []
        
        for metric in metrics:
            metric_df = dataset_df[dataset_df['Metric'] == metric]
            if len(metric_df) > 0:
                baseline_vals.append(metric_df['Baseline'].values[0])
                optimized_vals.append(metric_df['Optimized'].values[0])
        
        baseline_vals += baseline_vals[:1]
        optimized_vals += optimized_vals[:1]
        
        ax.plot(angles, baseline_vals, 'o-', linewidth=2, label='Baseline', 
               color=COLORS['baseline'])
        ax.fill(angles, baseline_vals, alpha=0.15, color=COLORS['baseline'])
        
        ax.plot(angles, optimized_vals, 'o-', linewidth=2, label='RankSaga Optimized', 
               color=COLORS['optimized'])
        ax.fill(angles, optimized_vals, alpha=0.15, color=COLORS['optimized'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(dataset.upper(), fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'radar_charts.png', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved radar charts")
    plt.close()


def plot_domain_comparison(results: Dict[str, Dict[str, Any]]):
    """Create domain-specific model comparison chart."""
    if 'baseline' not in results or 'scientific' not in results or 'general' not in results:
        logger.warning("Missing domain-specific results")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Domain-Specific Model Performance Comparison', 
                 fontsize=16, fontweight='bold', y=1.0)
    
    baseline_data = results['baseline'].get('results', {})
    scientific_data = results['scientific'].get('results', {})
    general_data = results['general'].get('results', {})
    
    # Scientific domain comparison
    ax1 = axes[0]
    scientific_datasets = ['scifact', 'nfcorpus', 'scidocs']
    metrics = ['NDCG@10']
    
    x = np.arange(len(scientific_datasets))
    width = 0.25
    
    baseline_vals = [baseline_data.get(d, {}).get('NDCG@10', 0) for d in scientific_datasets]
    scientific_vals = [scientific_data.get(d, {}).get('NDCG@10', 0) for d in scientific_datasets]
    
    ax1.bar(x - width, baseline_vals, width, label='Baseline', color=COLORS['baseline'], alpha=0.8)
    ax1.bar(x, scientific_vals, width, label='Scientific Domain Model', 
           color=COLORS['secondary'], alpha=0.8)
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('NDCG@10')
    ax1.set_title('Scientific Domain Models')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scientific_datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # General domain comparison
    ax2 = axes[1]
    general_datasets = ['quora']
    
    baseline_quora = baseline_data.get('quora', {}).get('NDCG@10', 0)
    general_quora = general_data.get('quora', {}).get('NDCG@10', 0)
    
    x = np.arange(1)
    ax2.bar(x - width/2, [baseline_quora], width, label='Baseline', 
           color=COLORS['baseline'], alpha=0.8)
    ax2.bar(x + width/2, [general_quora], width, label='General Domain Model', 
           color=COLORS['accent'], alpha=0.8)
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('General Domain Models')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['quora'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_comparison.png', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved domain comparison chart")
    plt.close()


def plot_metric_trends(df: pd.DataFrame):
    """Create line charts showing performance across different metrics."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    datasets = df['Dataset'].unique()
    metrics = ['NDCG@10', 'NDCG@100', 'MAP@100', 'Recall@100']
    x_pos = np.arange(len(metrics))
    
    width = 0.15
    for idx, dataset in enumerate(datasets):
        dataset_df = df[df['Dataset'] == dataset]
        baseline_vals = [dataset_df[dataset_df['Metric'] == m]['Baseline'].values[0] 
                        if len(dataset_df[dataset_df['Metric'] == m]) > 0 else 0 
                        for m in metrics]
        optimized_vals = [dataset_df[dataset_df['Metric'] == m]['Optimized'].values[0] 
                         if len(dataset_df[dataset_df['Metric'] == m]) > 0 else 0 
                         for m in metrics]
        
        offset = (idx - len(datasets)/2) * width + width/2
        ax.plot(x_pos + offset, baseline_vals, 'o--', linewidth=2, 
               label=f'{dataset} Baseline', alpha=0.7, markersize=6)
        ax.plot(x_pos + offset, optimized_vals, 's-', linewidth=2, 
               label=f'{dataset} Optimized', alpha=0.9, markersize=6)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Trends Across Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metric_trends.png', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved metric trends chart")
    plt.close()


def plot_summary_statistics(df: pd.DataFrame):
    """Create summary statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('RankSaga Optimization: Summary Statistics', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Average improvement by metric
    ax1 = axes[0, 0]
    metric_avg = df.groupby('Metric')['Improvement_%'].mean().sort_values(ascending=True)
    colors = [COLORS['improvement'] if x > 0 else COLORS['regression'] for x in metric_avg.values]
    bars = ax1.barh(metric_avg.index, metric_avg.values, color=colors, alpha=0.8)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Average Improvement (%)')
    ax1.set_title('Average Improvement by Metric')
    ax1.grid(axis='x', alpha=0.3)
    for i, (idx, val) in enumerate(metric_avg.items()):
        ax1.text(val + (0.5 if val > 0 else -0.5), i, f'{val:.2f}%',
               va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    # Improvement distribution
    ax2 = axes[0, 1]
    improvements = df['Improvement_%'].values
    ax2.hist(improvements, bins=20, color=COLORS['optimized'], alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax2.axvline(x=np.mean(improvements), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(improvements):.2f}%')
    ax2.set_xlabel('Improvement (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Improvement Distribution')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Datasets improved vs regressed
    ax3 = axes[1, 0]
    ndcg10_df = df[df['Metric'] == 'NDCG@10']
    improved = (ndcg10_df['Improvement_%'] > 0).sum()
    regressed = (ndcg10_df['Improvement_%'] < 0).sum()
    ax3.bar(['Improved', 'Regressed'], [improved, regressed], 
           color=[COLORS['improvement'], COLORS['regression']], alpha=0.8)
    ax3.set_ylabel('Number of Datasets')
    ax3.set_title('NDCG@10: Datasets Improved vs Regressed')
    ax3.grid(axis='y', alpha=0.3)
    for i, (label, val) in enumerate(zip(['Improved', 'Regressed'], [improved, regressed])):
        ax3.text(i, val + 0.1, str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Absolute improvement by dataset
    ax4 = axes[1, 1]
    dataset_abs = df.groupby('Dataset')['Absolute_Improvement'].sum().sort_values(ascending=True)
    colors = [COLORS['improvement'] if x > 0 else COLORS['regression'] for x in dataset_abs.values]
    bars = ax4.barh(dataset_abs.index, dataset_abs.values, color=colors, alpha=0.8)
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Total Absolute Improvement (sum across metrics)')
    ax4.set_title('Overall Improvement by Dataset')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_statistics.png', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved summary statistics chart")
    plt.close()


def main():
    """Generate all visualizations."""
    logger.info("Loading results data...")
    results = load_all_results()
    
    if not results:
        logger.error("No results found. Please ensure result files exist.")
        return
    
    logger.info("Creating comparison DataFrame...")
    df = create_comparison_dataframe(results)
    
    if df.empty:
        logger.error("Failed to create comparison DataFrame.")
        return
    
    logger.info(f"Generating visualizations in {OUTPUT_DIR}...")
    
    # Generate all visualizations
    plot_main_comparison(df)
    plot_improvement_heatmap(df)
    plot_improvement_percentages(df)
    plot_scatter_comparison(df)
    plot_radar_charts(df)
    plot_domain_comparison(results)
    plot_metric_trends(df)
    plot_summary_statistics(df)
    
    logger.info("✅ All visualizations generated successfully!")
    logger.info(f"Charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

