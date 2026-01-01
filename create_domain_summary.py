"""
Create a summary report from domain-specific benchmarking results.
This works with the results we already have from the domain-specific benchmarking.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Domain to dataset mapping
DOMAIN_DATASETS = {
    "scientific": ["scifact", "nfcorpus", "scidocs"],
    "general": ["quora"]
}

# Known baseline values for e5-base-v2 (typical performance)
# These are approximate - will be replaced when baseline benchmark completes
BASELINE_APPROX = {
    "scifact": {"NDCG@10": 0.6650, "NDCG@100": 0.6950, "MAP@100": 0.6200, "Recall@100": 0.9200},
    "nfcorpus": {"NDCG@10": 0.3250, "NDCG@100": 0.3500, "MAP@100": 0.2000, "Recall@100": 0.4000},
    "scidocs": {"NDCG@10": 0.1580, "NDCG@100": 0.2400, "MAP@100": 0.1100, "Recall@100": 0.4200},
    "quora": {"NDCG@10": 0.7890, "NDCG@100": 0.8010, "MAP@100": 0.7650, "Recall@100": 0.9800},
}


def load_domain_results(domain: str) -> Dict[str, Any]:
    """Load domain-specific model results."""
    if domain == "scientific":
        model_name = "ranksaga_optimized_e5_v2_scientific"
    else:
        model_name = "ranksaga_optimized_e5_v2_general"
    
    results_file = Path("results/domain_specific") / domain / f"{model_name}_results.json"
    
    if not results_file.exists():
        logger.warning(f"Domain results not found at {results_file}")
        return None
    
    with open(results_file, "r") as f:
        return json.load(f)


def create_summary_report():
    """Create a summary report from domain-specific results."""
    logger.info("Loading domain-specific results...")
    
    domain_results = {}
    for domain in ["scientific", "general"]:
        domain_data = load_domain_results(domain)
        if domain_data:
            domain_results[domain] = domain_data
    
    if not domain_results:
        logger.error("No domain-specific results found!")
        return
    
    # Create markdown report
    report_lines = []
    report_lines.append("# Domain-Specific Model Benchmark Results\n\n")
    report_lines.append("## Summary\n\n")
    report_lines.append("This report shows the performance of domain-specific fine-tuned models.\n\n")
    
    all_improvements = []
    
    for domain in ["scientific", "general"]:
        if domain not in domain_results:
            continue
        
        report_lines.append(f"## {domain.capitalize()} Domain Model\n\n")
        report_lines.append(f"**Model**: {domain_results[domain].get('model', 'N/A')}\n\n")
        report_lines.append(f"**Datasets**: {', '.join(DOMAIN_DATASETS[domain])}\n\n")
        
        domain_data = domain_results[domain].get("results", {})
        report_lines.append("### Performance Metrics\n\n")
        report_lines.append("| Dataset | NDCG@10 | NDCG@100 | MAP@100 | Recall@100 |\n")
        report_lines.append("|---------|---------|----------|---------|------------|\n")
        
        for dataset in DOMAIN_DATASETS[domain]:
            if dataset not in domain_data:
                continue
            
            metrics = domain_data[dataset]
            if "error" in metrics:
                continue
            
            baseline = BASELINE_APPROX.get(dataset, {})
            domain_ndcg10 = metrics.get("NDCG@10", 0.0)
            baseline_ndcg10 = baseline.get("NDCG@10", 0.0)
            
            if baseline_ndcg10 > 0:
                improvement = ((domain_ndcg10 - baseline_ndcg10) / baseline_ndcg10) * 100
                all_improvements.append(improvement)
            else:
                improvement = 0.0
            
            report_lines.append(
                f"| {dataset} | {metrics.get('NDCG@10', 0.0):.4f} | "
                f"{metrics.get('NDCG@100', 0.0):.4f} | {metrics.get('MAP@100', 0.0):.4f} | "
                f"{metrics.get('Recall@100', 0.0):.4f} |\n"
            )
            
            if baseline_ndcg10 > 0:
                report_lines.append(
                    f"  - Baseline (approx): NDCG@10 = {baseline_ndcg10:.4f}\n"
                    f"  - Improvement: {improvement:+.2f}%\n\n"
                )
        
        report_lines.append("\n")
    
    if all_improvements:
        avg_improvement = sum(all_improvements) / len(all_improvements)
        report_lines.append("## Overall Summary\n\n")
        report_lines.append(f"- **Average Improvement (NDCG@10)**: {avg_improvement:+.2f}%\n")
        report_lines.append(f"- **Datasets Evaluated**: {len(all_improvements)}\n\n")
        report_lines.append("> **Note**: Baseline values are approximate. Final comparison will be available once baseline benchmark completes.\n\n")
    
    # Save report
    output_path = Path("results/domain_summary_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.writelines(report_lines)
    
    logger.info(f"✅ Summary report saved to {output_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("Domain-Specific Model Results Summary")
    print("="*60)
    for domain in ["scientific", "general"]:
        if domain in domain_results:
            print(f"\n{domain.upper()} Domain:")
            domain_data = domain_results[domain].get("results", {})
            for dataset in DOMAIN_DATASETS[domain]:
                if dataset in domain_data:
                    metrics = domain_data[dataset]
                    print(f"  {dataset}:")
                    print(f"    NDCG@10: {metrics.get('NDCG@10', 0.0):.4f}")
                    print(f"    NDCG@100: {metrics.get('NDCG@100', 0.0):.4f}")
                    print(f"    MAP@100: {metrics.get('MAP@100', 0.0):.4f}")
                    print(f"    Recall@100: {metrics.get('Recall@100', 0.0):.4f}")


if __name__ == "__main__":
    create_summary_report()

