"""
Modal.com deployment for domain-specific BEIR benchmarking.
Benchmarks domain-specific models on their respective domains.
"""
import modal

# Create a Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "beir>=2.2.0",
        "sentence-transformers>=3.0.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
        "scipy>=1.7.0",
        "tabulate>=0.9.0",
    )
)

app = modal.App("ranksaga-domain-benchmark", image=image)

# Volumes
results_volume = modal.Volume.from_name("ranksaga-benchmark-results", create_if_missing=True)
models_volume = modal.Volume.from_name("ranksaga-models", create_if_missing=True)

# Domain to dataset mapping
DOMAIN_DATASETS = {
    "scientific": ["scifact", "nfcorpus", "scidocs"],
    "general": ["quora"]
}


@app.function(
    image=image,
    gpu="H100",  # Use H100 GPU for faster benchmarking
    timeout=7200,
    volumes={"/results": results_volume, "/models": models_volume},
    memory=32768,
)
def benchmark_domain_model(
    domain: str = "scientific",
    model_name: str = None,
):
    """
    Benchmark a domain-specific model on its domain datasets.
    
    Args:
        domain: Domain to benchmark ("scientific" or "general")
        model_name: Name of the model (defaults to domain-specific model name)
    """
    import logging
    import json
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from tqdm import tqdm
    import os
    import torch
    from datetime import datetime
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    if domain not in DOMAIN_DATASETS:
        return {"status": "error", "message": f"Invalid domain: {domain}"}
    
    datasets = DOMAIN_DATASETS[domain]
    
    # Default model names
    if model_name is None:
        if domain == "scientific":
            model_name = "ranksaga-optimized-e5-v2-scientific"
        else:
            model_name = "ranksaga-optimized-e5-v2-general"
    
    EVALUATION_METRICS = [1, 3, 5, 10, 100, 1000]
    
    # Create directories (use /results for datasets too)
    datasets_dir = "/results/datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    results_dir = Path(f"/results/domain_specific/{domain}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ GPU not available")
    
    # Load model
    model_path = f"/models/{model_name}"
    logger.info(f"Loading {domain} domain model from: {model_path}")
    
    try:
        model = SentenceTransformer(model_path)
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return {"status": "error", "message": f"Could not load model: {e}"}
    
    def manual_retrieve(model, corpus, queries, top_k=1000):
        """Manually perform dense retrieval using the model."""
        logger.info("Encoding documents...")
        doc_texts = [corpus[doc_id]["text"] for doc_id in corpus.keys()]
        doc_ids = list(corpus.keys())
        
        batch_size = 256 if torch.cuda.is_available() else 32
        doc_embeddings = model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        logger.info("Encoding queries and retrieving...")
        query_texts = list(queries.values())
        query_ids = list(queries.keys())
        query_embeddings = model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        logger.info("Computing similarities...")
        scores = np.dot(doc_embeddings, query_embeddings.T)
        
        results_dict = {}
        for i, query_id in enumerate(tqdm(query_ids, desc="Building results")):
            top_indices = np.argsort(scores[:, i])[::-1][:top_k]
            results_dict[query_id] = {
                doc_ids[idx]: float(scores[idx, i]) for idx in top_indices
            }
        
        return results_dict
    
    def load_beir_dataset(dataset_name, datasets_dir, split="test"):
        """Load a BEIR dataset."""
        dataset_path = Path(datasets_dir) / dataset_name
        
        if not dataset_path.exists():
            logger.info(f"Downloading dataset {dataset_name}...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, str(datasets_dir))
            dataset_path = Path(data_path)
        
        logger.info(f"Loading {dataset_name} {split} split...")
        corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split=split)
        logger.info(f"Loaded {dataset_name}: {len(corpus)} documents, {len(queries)} queries")
        return corpus, queries, qrels
    
    all_results = {}
    
    logger.info("="*60)
    logger.info(f"Benchmarking {domain.upper()} Domain Model")
    logger.info(f"Model: {model_name}")
    logger.info(f"Datasets: {datasets}")
    logger.info("="*60)
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            corpus, queries, qrels = load_beir_dataset(dataset_name, datasets_dir, split="test")
            results_dict = manual_retrieve(model, corpus, queries, top_k=max(EVALUATION_METRICS))
            
            evaluator = EvaluateRetrieval(k_values=EVALUATION_METRICS)
            ndcg, _map, recall, precision = evaluator.evaluate(qrels, results_dict, EVALUATION_METRICS)
            
            dataset_results = {
                "NDCG@10": ndcg.get("NDCG@10", 0.0),
                "NDCG@100": ndcg.get("NDCG@100", 0.0),
                "MAP@100": _map.get("MAP@100", 0.0),
                "Recall@100": recall.get("Recall@100", 0.0),
                "Precision@100": precision.get("P@100", 0.0),
            }
            all_results[dataset_name] = dataset_results
            
            logger.info(
                f"{dataset_name} Results:\n"
                f"  NDCG@10: {dataset_results['NDCG@10']:.4f}\n"
                f"  NDCG@100: {dataset_results['NDCG@100']:.4f}\n"
                f"  MAP@100: {dataset_results['MAP@100']:.4f}\n"
                f"  Recall@100: {dataset_results['Recall@100']:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {"error": str(e)}
            continue
    
    # Save results
    results_file = results_dir / f"{model_name.replace('-', '_')}_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "model": model_name,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "results": all_results
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {results_file}")
    results_volume.commit()
    logger.info("✅ Volume committed - results are persisted")
    
    return {
        "status": "success",
        "domain": domain,
        "model": model_name,
        "datasets_evaluated": list(all_results.keys()),
        "results": all_results,
        "results_file": str(results_file)
    }


@app.local_entrypoint()
def benchmark_all_domains():
    """Benchmark all domain-specific models."""
    import json
    
    results = {}
    
    for domain in DOMAIN_DATASETS.keys():
        print(f"\n{'='*60}")
        print(f"Benchmarking {domain.upper()} Domain")
        print(f"{'='*60}\n")
        
        result = benchmark_domain_model.remote(domain=domain)
        results[domain] = result
    
    print("\n" + "="*60)
    print("All Domain Benchmarks Summary")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    return results

