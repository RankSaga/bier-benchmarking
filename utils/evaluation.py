"""
Evaluation metrics and helper functions.
"""
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def format_metrics(ndcg: Dict, _map: Dict, recall: Dict, precision: Dict) -> Dict[str, float]:
    """
    Format evaluation metrics into a clean dictionary.
    
    Args:
        ndcg: NDCG metrics dictionary
        _map: MAP metrics dictionary
        recall: Recall metrics dictionary
        precision: Precision metrics dictionary
        
    Returns:
        Formatted metrics dictionary
    """
    return {
        "NDCG@10": ndcg.get("NDCG@10", 0.0),
        "NDCG@100": ndcg.get("NDCG@100", 0.0),
        "MAP@100": _map.get("MAP@100", 0.0),
        "Recall@100": recall.get("Recall@100", 0.0),
        "Precision@100": precision.get("P@100", 0.0),
    }


def save_results(
    results: Dict[str, Any],
    output_path: Path,
    model_name: str
) -> None:
    """
    Save benchmark results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Directory to save results
        model_name: Name of the model
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename
    safe_name = model_name.replace("/", "_").replace("-", "_")
    filename = output_path / f"{safe_name}_results.json"
    
    # Add metadata
    results_with_metadata = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    with open(filename, "w") as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print(f"Results saved to {filename}")


def load_results(results_path: Path) -> Dict[str, Any]:
    """
    Load benchmark results from JSON file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        Results dictionary
    """
    with open(results_path, "r") as f:
        return json.load(f)


def calculate_average_metrics(results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Calculate average metrics across multiple datasets.
    
    Args:
        results: Dictionary mapping dataset names to metrics
        
    Returns:
        Dictionary with average metrics
    """
    if not results:
        return {}
    
    metrics = ["NDCG@10", "NDCG@100", "MAP@100", "Recall@100", "Precision@100"]
    averages = {}
    
    for metric in metrics:
        values = [
            dataset_results.get(metric, 0.0)
            for dataset_results in results.values()
            if metric in dataset_results
        ]
        if values:
            averages[f"Average_{metric}"] = sum(values) / len(values)
    
    return averages

