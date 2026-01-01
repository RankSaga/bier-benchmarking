"""
Compare baseline vs optimized model performance.
"""
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

from config import (
    BASE_MODEL,
    OPTIMIZED_MODEL_NAME,
    PATHS,
    BEIR_DATASETS
)
from utils.evaluation import load_results, calculate_average_metrics

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_baseline_results() -> Dict[str, Any]:
    """Load baseline model results."""
    baseline_safe_name = BASE_MODEL.replace("/", "_")
    
    # Try multiple possible filenames (Modal vs local)
    possible_files = [
        PATHS["baseline_results"] / f"{baseline_safe_name}_results.json",
        PATHS["baseline_results"] / "e5_base_v2_results.json",  # Modal format
        PATHS["baseline_results"] / "intfloat_e5_base_v2_results.json",
    ]
    
    for baseline_file in possible_files:
        if baseline_file.exists():
            logger.info(f"Loading baseline results from {baseline_file}")
            return load_results(baseline_file)
    
    logger.error(f"Baseline results not found. Tried: {possible_files}")
    logger.info("Run run_baseline.py or download from Modal first")
    return None


def load_optimized_results() -> Dict[str, Any]:
    """Load optimized model results."""
    optimized_safe_name = OPTIMIZED_MODEL_NAME.replace("-", "_")
    
    # Try multiple possible filenames (Modal vs local)
    possible_files = [
        PATHS["optimized_results"] / f"{optimized_safe_name}_results.json",
        PATHS["optimized_results"] / "ranksaga_optimized_e5_v1_results.json",  # Modal format
    ]
    
    for optimized_file in possible_files:
        if optimized_file.exists():
            logger.info(f"Loading optimized results from {optimized_file}")
            return load_results(optimized_file)
    
    logger.error(f"Optimized results not found. Tried: {possible_files}")
    logger.info("Run run_optimized.py or download from Modal first")
    return None


def create_comparison_table(
    baseline_data: Dict[str, Any],
    optimized_data: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create a comparison DataFrame.
    
    Args:
        baseline_data: Baseline results dictionary
        optimized_data: Optimized results dictionary
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    baseline_results = baseline_data.get("results", {})
    optimized_results = optimized_data.get("results", {})
    
    for dataset in BEIR_DATASETS:
        if dataset not in baseline_results or dataset not in optimized_results:
            continue
        
        baseline_metrics = baseline_results[dataset]
        optimized_metrics = optimized_results[dataset]
        
        # Skip if there were errors
        if "error" in baseline_metrics or "error" in optimized_metrics:
            continue
        
        for metric in ["NDCG@10", "NDCG@100", "MAP@100", "Recall@100"]:
            baseline_value = baseline_metrics.get(metric, 0.0)
            optimized_value = optimized_metrics.get(metric, 0.0)
            
            if baseline_value > 0:
                improvement = ((optimized_value - baseline_value) / baseline_value) * 100
            else:
                improvement = 0.0
            
            comparison_data.append({
                "Dataset": dataset,
                "Metric": metric,
                "Baseline": baseline_value,
                "Optimized": optimized_value,
                "Improvement_%": improvement,
                "Absolute_Improvement": optimized_value - baseline_value
            })
    
    df = pd.DataFrame(comparison_data)
    return df


def visualize_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Create visualization of improvements.
    
    Args:
        df: Comparison DataFrame
        output_dir: Directory to save visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to NDCG@10 for main comparison
    ndcg10_df = df[df["Metric"] == "NDCG@10"].copy()
    
    if len(ndcg10_df) == 0:
        logger.warning("No NDCG@10 data to visualize")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Improvement percentages by dataset
    ax1 = axes[0]
    ndcg10_df_sorted = ndcg10_df.sort_values("Improvement_%", ascending=True)
    ax1.barh(
        ndcg10_df_sorted["Dataset"],
        ndcg10_df_sorted["Improvement_%"],
        color="steelblue"
    )
    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Improvement (%)")
    ax1.set_title("RankSaga Optimization Improvements (NDCG@10)")
    ax1.grid(axis="x", alpha=0.3)
    
    # Plot 2: Baseline vs Optimized scatter
    ax2 = axes[1]
    ax2.scatter(
        ndcg10_df["Baseline"],
        ndcg10_df["Optimized"],
        s=100,
        alpha=0.7,
        color="steelblue"
    )
    
    # Add diagonal line (no improvement)
    max_val = max(
        ndcg10_df["Baseline"].max(),
        ndcg10_df["Optimized"].max()
    )
    min_val = min(
        ndcg10_df["Baseline"].min(),
        ndcg10_df["Optimized"].min()
    )
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        alpha=0.5,
        label="No improvement"
    )
    
    # Add dataset labels
    for _, row in ndcg10_df.iterrows():
        ax2.annotate(
            row["Dataset"],
            (row["Baseline"], row["Optimized"]),
            fontsize=8,
            alpha=0.7
        )
    
    ax2.set_xlabel("Baseline NDCG@10")
    ax2.set_ylabel("Optimized NDCG@10")
    ax2.set_title("Baseline vs Optimized Performance")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    viz_path = output_dir / "comparison_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    logger.info(f"Visualization saved to {viz_path}")
    plt.close()


def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Generate summary statistics.
    
    Args:
        df: Comparison DataFrame
        
    Returns:
        Dictionary of summary statistics
    """
    ndcg10_df = df[df["Metric"] == "NDCG@10"]
    
    if len(ndcg10_df) == 0:
        return {}
    
    return {
        "average_improvement_%": ndcg10_df["Improvement_%"].mean(),
        "max_improvement_%": ndcg10_df["Improvement_%"].max(),
        "min_improvement_%": ndcg10_df["Improvement_%"].min(),
        "average_absolute_improvement": ndcg10_df["Absolute_Improvement"].mean(),
        "datasets_improved": (ndcg10_df["Improvement_%"] > 0).sum(),
        "total_datasets": len(ndcg10_df)
    }


def generate_markdown_report(
    df: pd.DataFrame,
    summary_stats: Dict[str, float],
    output_dir: Path
):
    """
    Generate a markdown report.
    
    Args:
        df: Comparison DataFrame
        summary_stats: Summary statistics
        output_dir: Directory to save report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"
    
    with open(report_path, "w") as f:
        f.write("# RankSaga Model Optimization Benchmark Results\n\n")
        f.write("## Summary\n\n")
        
        if summary_stats:
            f.write(f"- **Average Improvement**: {summary_stats.get('average_improvement_%', 0):.2f}%\n")
            f.write(f"- **Maximum Improvement**: {summary_stats.get('max_improvement_%', 0):.2f}%\n")
            f.write(f"- **Minimum Improvement**: {summary_stats.get('min_improvement_%', 0):.2f}%\n")
            f.write(f"- **Datasets Improved**: {summary_stats.get('datasets_improved', 0)}/{summary_stats.get('total_datasets', 0)}\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("### NDCG@10 Comparison\n\n")
        
        ndcg10_df = df[df["Metric"] == "NDCG@10"].copy()
        if len(ndcg10_df) > 0:
            f.write("| Dataset | Baseline | Optimized | Improvement % |\n")
            f.write("|---------|----------|-----------|---------------|\n")
            for _, row in ndcg10_df.iterrows():
                f.write(
                    f"| {row['Dataset']} | {row['Baseline']:.4f} | "
                    f"{row['Optimized']:.4f} | {row['Improvement_%']:.2f}% |\n"
                )
        
        f.write("\n### All Metrics Comparison\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except AttributeError:
            # Fallback if tabulate is not available
            f.write(df.to_string(index=False))
    
    logger.info(f"Report saved to {report_path}")


def compare_results():
    """Main function to compare baseline and optimized results."""
    logger.info("Loading baseline and optimized results...")
    
    baseline_data = load_baseline_results()
    optimized_data = load_optimized_results()
    
    if baseline_data is None or optimized_data is None:
        logger.error("Cannot compare results - missing baseline or optimized data")
        return
    
    logger.info("Creating comparison table...")
    df = create_comparison_table(baseline_data, optimized_data)
    
    if df.empty:
        logger.error("Comparison table is empty. Check that results exist for both models.")
        return
    
    # Save comparison table
    comparison_csv = PATHS["results"] / "comparison_table.csv"
    df.to_csv(comparison_csv, index=False)
    logger.info(f"Comparison table saved to {comparison_csv}")
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(df)
    logger.info("\nSummary Statistics:")
    for key, value in summary_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualize_comparison(df, PATHS["results"])
    
    # Generate markdown report
    logger.info("Generating markdown report...")
    generate_markdown_report(df, summary_stats, PATHS["results"])
    
    logger.info("\n" + "="*60)
    logger.info("Comparison complete!")
    logger.info("="*60)
    logger.info(f"Results saved to: {PATHS['results']}")


if __name__ == "__main__":
    compare_results()

