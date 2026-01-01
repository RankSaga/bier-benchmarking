"""
Modal.com deployment for BEIR benchmarking.
Run: modal deploy modal_app.py
"""
import modal

# Create a Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "beir>=2.2.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "tabulate>=0.9.0",
    )
)

app = modal.App("ranksaga-beir-benchmark", image=image)

# Volume for persistent storage of results
results_volume = modal.Volume.from_name("ranksaga-benchmark-results", create_if_missing=True)

# Volume for models (shared with fine-tuning app)
models_volume = modal.Volume.from_name("ranksaga-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # Use A10G GPU for fastest execution
    timeout=7200,  # 2 hour timeout
    volumes={"/results": results_volume},
    memory=32768,  # 32GB RAM for large datasets
)
def run_benchmark(dataset_name: str = None):
    """
    Run BEIR benchmarking on Modal with GPU acceleration.
    
    Args:
        dataset_name: Specific dataset to run, or None for all datasets
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
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    BASE_MODEL = "intfloat/e5-base-v2"
    BEIR_DATASETS = ["scifact", "nfcorpus", "scidocs", "quora"] if dataset_name is None else [dataset_name]
    EVALUATION_METRICS = [1, 3, 5, 10, 100, 1000]
    
    # Create directories
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("/results/baseline", exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠️ GPU not available, using CPU (slower)")
    
    def manual_retrieve(model, corpus, queries, top_k=1000):
        """Manually perform dense retrieval using the model with GPU acceleration."""
        logger.info("Encoding documents...")
        doc_texts = [corpus[doc_id]["text"] for doc_id in corpus.keys()]
        doc_ids = list(corpus.keys())
        
        # Use larger batch size for GPU
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
        
        # Compute all similarities at once (much faster on GPU)
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
    
    # Load model
    logger.info(f"Loading baseline model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)
    if torch.cuda.is_available():
        # Move model to GPU if available
        model = model.to("cuda")
    
    all_results = {}
    
    for dataset_name in BEIR_DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            corpus, queries, qrels = load_beir_dataset(dataset_name, "datasets", split="test")
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
    
    # Save results to persistent volume
    results_file = "/results/baseline/e5_base_v2_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "model": BASE_MODEL,
            "results": all_results
        }, f, indent=2)
    
    # Commit volume to persist results
    results_volume.commit()
    
    logger.info(f"\n✅ Results saved to {results_file}")
    logger.info("✅ Volume committed - results are persisted")
    
    # Return results summary
    return {
        "status": "success",
        "model": BASE_MODEL,
        "datasets_evaluated": list(all_results.keys()),
        "results": all_results,
        "results_file": results_file
    }


@app.function(
    image=image,
    volumes={"/results": results_volume},
)
def get_results():
    """Retrieve results from Modal volume."""
    import json
    results_file = "/results/baseline/e5_base_v2_results.json"
    try:
        with open(results_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Results file not found. Run benchmark first."}


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/results": results_volume, "/models": models_volume},
    memory=32768,
)
def run_optimized_benchmark(model_name: str = "ranksaga-optimized-e5-v2", dataset_name: str = None):
    """
    Benchmark the fine-tuned optimized model on BEIR datasets.
    
    Args:
        model_name: Name of the fine-tuned model (should be in /models volume)
        dataset_name: Specific dataset to run, or None for all datasets
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
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    BEIR_DATASETS = ["scifact", "nfcorpus", "scidocs", "quora"] if dataset_name is None else [dataset_name]
    EVALUATION_METRICS = [1, 3, 5, 10, 100, 1000]
    
    # Create directories
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("/results/optimized", exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ GPU not available")
    
    # Load model from volume
    model_path = f"/models/{model_name}"
    logger.info(f"Loading optimized model from: {model_path}")
    
    try:
        model = SentenceTransformer(model_path)
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        logger.info("Make sure you have run fine_tune_model first")
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
    
    for dataset_name in BEIR_DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating optimized model on dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            corpus, queries, qrels = load_beir_dataset(dataset_name, "datasets", split="test")
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
    
    # Save results to persistent volume
    results_file = f"/results/optimized/{model_name.replace('-', '_')}_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "model": model_name,
            "results": all_results
        }, f, indent=2)
    
    # Commit volume to persist results
    results_volume.commit()
    
    logger.info(f"\n✅ Results saved to {results_file}")
    logger.info("✅ Volume committed - results are persisted")
    
    return {
        "status": "success",
        "model": model_name,
        "datasets_evaluated": list(all_results.keys()),
        "results": all_results,
        "results_file": results_file
    }


@app.local_entrypoint()
def main(dataset_name: str = None):
    """Run benchmark via CLI: modal run modal_app.py --dataset-name scifact"""
    import json
    result = run_benchmark.remote(dataset_name=dataset_name)
    print("\n" + "="*60)
    print("Benchmark Results Summary")
    print("="*60)
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def benchmark_optimized(model_name: str = "ranksaga-optimized-e5-v2", dataset_name: str = None):
    """Run optimized benchmark via CLI: modal run modal_app.py::benchmark_optimized"""
    import json
    result = run_optimized_benchmark.remote(model_name=model_name, dataset_name=dataset_name)
    print("\n" + "="*60)
    print("Optimized Benchmark Results Summary")
    print("="*60)
    print(json.dumps(result, indent=2))
    return result

