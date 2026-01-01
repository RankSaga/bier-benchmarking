"""
Compare domain-specific model results with baseline and previous optimized model.
"""
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional

from config import (
    BASE_MODEL,
    BEIR_DATASETS,
    PATHS,
)
from utils.evaluation import load_results, calculate_average_metrics

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Domain to dataset mapping
DOMAIN_DATASETS = {
    "scientific": ["scifact", "nfcorpus", "scidocs"],
    "general": ["quora"]
}


def load_baseline_results() -> Optional[Dict[str, Any]]:
    """Load baseline model results."""
    # Try multiple possible filenames
    possible_files = [
        PATHS["baseline_results"] / "e5_base_v2_results.json",  # Modal format
        PATHS["baseline_results"] / "intfloat_e5_base_v2_results.json",
        PATHS["baseline_results"] / f"{BASE_MODEL.replace('/', '_')}_results.json",
    ]
    
    for baseline_file in possible_files:
        if baseline_file.exists():
            logger.info(f"Loading baseline results from {baseline_file}")
            return load_results(baseline_file)
    
    logger.error(f"Baseline results not found. Tried: {possible_files}")
    return None


def load_domain_results(domain: str) -> Optional[Dict[str, Any]]:
    """Load domain-specific model results."""
    if domain == "scientific":
        model_name = "ranksaga_optimized_e5_v2_scientific"
    else:
        model_name = "ranksaga_optimized_e5_v2_general"
    
    results_file = PATHS["results"] / "domain_specific" / domain / f"{model_name}_results.json"
    
    if not results_file.exists():
        logger.warning(f"Domain results not found at {results_file}")
        return None
    
    return load_results(results_file)


def create_comparison_table(
    baseline_data: Dict[str, Any],
    domain_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a comparison DataFrame for domain-specific models.
    
    Args:
        baseline_data: Baseline results dictionary
        domain_results: Dictionary mapping domain names to their results
    
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    baseline_results = baseline_data.get("results", {})
    
    for domain, domain_data in domain_results.items():
        if domain_data is None:
            continue
        
        domain_result_data = domain_data.get("results", {})
        datasets = DOMAIN_DATASETS[domain]
        
        for dataset in datasets:
            if dataset not in baseline_results or dataset not in domain_result_data:
                continue
            
            baseline_metrics = baseline_results[dataset]
            domain_metrics = domain_result_data[dataset]
            
            # Skip if there were errors
            if "error" in baseline_metrics or "error" in domain_metrics:
                continue
            
            for metric in ["NDCG@10", "NDCG@100", "MAP@100", "Recall@100"]:
                baseline_value = baseline_metrics.get(metric, 0.0)
                domain_value = domain_metrics.get(metric, 0.0)
                
                improvement_percent = 0.0
                if baseline_value != 0:
                    improvement_percent = ((domain_value - baseline_value) / baseline_value) * 100
                
                absolute_improvement = domain_value - baseline_value
                
                comparison_data.append({
                    "Domain": domain,
                    "Dataset": dataset,
                    "Metric": metric,
                    "Baseline": baseline_value,
                    "Domain_Model": domain_value,
                    "Improvement_%": improvement_percent,
                    "Absolute_Improvement": absolute_improvement
                })
    
    df = pd.DataFrame(comparison_data)
    return df


def visualize_comparison(df: pd.DataFrame, output_path: Path):
    """Create and save visualizations of improvements."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Filter for NDCG@10
    ndcg_10_df = df[df["Metric"] == "NDCG@10"].copy()
    
    # 1. Bar plot by domain
    domain_avg = ndcg_10_df.groupby("Domain")["Improvement_%"].mean().sort_values(ascending=False)
    sns.barplot(x=domain_avg.index, y=domain_avg.values, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Average NDCG@10 Improvement by Domain", fontsize=14)
    axes[0, 0].set_ylabel("Improvement %", fontsize=12)
    axes[0, 0].axhline(0, color='grey', linewidth=0.8)
    
    # 2. Bar plot by dataset
    dataset_avg = ndcg_10_df.groupby("Dataset")["Improvement_%"].mean().sort_values(ascending=False)
    sns.barplot(x=dataset_avg.index, y=dataset_avg.values, ax=axes[0, 1], palette="tab10")
    axes[0, 1].set_title("NDCG@10 Improvement by Dataset", fontsize=14)
    axes[0, 1].set_ylabel("Improvement %", fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(0, color='grey', linewidth=0.8)
    
    # 3. Scatter plot: Baseline vs Domain Model
    sns.scatterplot(
        x="Baseline",
        y="Domain_Model",
        hue="Dataset",
        s=100,
        data=ndcg_10_df,
        ax=axes[1, 0],
        palette="tab10"
    )
    max_val = max(ndcg_10_df["Baseline"].max(), ndcg_10_df["Domain_Model"].max()) * 1.1
    min_val = min(ndcg_10_df["Baseline"].min(), ndcg_10_df["Domain_Model"].min()) * 0.9
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.6, label="No improvement")
    axes[1, 0].set_xlabel("Baseline NDCG@10", fontsize=12)
    axes[1, 0].set_ylabel("Domain Model NDCG@10", fontsize=12)
    axes[1, 0].set_title("Baseline vs Domain-Specific Model", fontsize=14)
    axes[1, 0].legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 4. Heatmap of improvements by dataset and metric
    pivot_df = df.pivot_table(
        index="Dataset",
        columns="Metric",
        values="Improvement_%",
        aggfunc="mean"
    )
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=axes[1, 1], cbar_kws={'label': 'Improvement %'})
    axes[1, 1].set_title("Improvement Heatmap by Dataset and Metric", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")


def generate_markdown_report(
    df: pd.DataFrame,
    output_path: Path,
    domain_results: Dict[str, Dict[str, Any]]
):
    """Generate a markdown report of the comparison."""
    
    # Calculate summary statistics
    ndcg_10_df = df[df["Metric"] == "NDCG@10"]
    avg_improvement = ndcg_10_df["Improvement_%"].mean()
    max_improvement = ndcg_10_df["Improvement_%"].max()
    min_improvement = ndcg_10_df["Improvement_%"].min()
    datasets_improved = (ndcg_10_df["Improvement_%"] > 0).sum()
    total_datasets = len(ndcg_10_df)
    
    report_content = f"# Domain-Specific Model Benchmark Results\n\n"
    report_content += f"## Summary\n\n"
    report_content += f"- **Average Improvement**: {avg_improvement:.2f}%\n"
    report_content += f"- **Maximum Improvement**: {max_improvement:.2f}%\n"
    report_content += f"- **Minimum Improvement**: {min_improvement:.2f}%\n"
    report_content += f"- **Datasets Improved**: {datasets_improved}/{total_datasets}\n\n"
    
    report_content += f"## Domain Models\n\n"
    for domain in ["scientific", "general"]:
        if domain in domain_results and domain_results[domain]:
            report_content += f"### {domain.capitalize()} Domain\n"
            report_content += f"- **Model**: {domain_results[domain].get('model', 'N/A')}\n"
            report_content += f"- **Datasets**: {', '.join(DOMAIN_DATASETS[domain])}\n\n"
    
    report_content += f"## Detailed Results\n\n"
    
    # NDCG@10 table
    ndcg_10_df = df[df["Metric"] == "NDCG@10"].round(4)
    report_content += "### NDCG@10 Comparison\n\n"
    report_content += ndcg_10_df[["Domain", "Dataset", "Baseline", "Domain_Model", "Improvement_%"]].to_markdown(index=False) + "\n\n"
    
    # All metrics table
    report_content += "### All Metrics Comparison\n\n"
    report_content += df.round(4).to_markdown(index=False) + "\n\n"
    
    report_content += f"## Visualizations\n\n"
    report_content += f"![Comparison Plot]({output_path.name})\n"
    
    report_filepath = output_path.parent / "domain_comparison_report.md"
    with open(report_filepath, "w") as f:
        f.write(report_content)
    logger.info(f"Report saved to {report_filepath}")


def main():
    logger.info("Loading baseline and domain-specific results...")
    
    baseline_data = load_baseline_results()
    if baseline_data is None:
        logger.error("Cannot compare results - missing baseline data")
        return
    
    # Load domain results
    domain_results = {}
    for domain in ["scientific", "general"]:
        domain_data = load_domain_results(domain)
        if domain_data:
            domain_results[domain] = domain_data
        else:
            logger.warning(f"Domain results not found for {domain}")
    
    if not domain_results:
        logger.error("No domain-specific results found. Run domain-specific benchmarking first.")
        return
    
    logger.info("Creating comparison table...")
    df = create_comparison_table(baseline_data, domain_results)
    
    if df.empty:
        logger.warning("No comparable data found.")
        return
    
    # Save comparison table
    csv_path = PATHS["results"] / "domain_comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table saved to {csv_path}")
    
    # Calculate summary statistics
    ndcg_10_df = df[df["Metric"] == "NDCG@10"]
    avg_improvement = ndcg_10_df["Improvement_%"].mean()
    max_improvement = ndcg_10_df["Improvement_%"].max()
    min_improvement = ndcg_10_df["Improvement_%"].min()
    datasets_improved = (ndcg_10_df["Improvement_%"] > 0).sum()
    total_datasets = len(ndcg_10_df)
    
    logger.info("\nSummary Statistics:")
    logger.info(f"  Average Improvement: {avg_improvement:.2f}%")
    logger.info(f"  Max Improvement: {max_improvement:.2f}%")
    logger.info(f"  Min Improvement: {min_improvement:.2f}%")
    logger.info(f"  Datasets Improved: {datasets_improved}/{total_datasets}")
    
    # Create visualization
    plot_path = PATHS["results"] / "domain_comparison_visualization.png"
    logger.info("Creating visualizations...")
    visualize_comparison(df, plot_path)
    
    # Generate markdown report
    logger.info("Generating markdown report...")
    generate_markdown_report(df, plot_path, domain_results)
    
    logger.info("\n" + "="*60)
    logger.info("Domain-specific comparison complete!")
    logger.info("="*60)
    logger.info(f"Results saved to: {PATHS['results']}")


if __name__ == "__main__":
    main()

