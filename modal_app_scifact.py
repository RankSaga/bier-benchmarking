"""
Modal.com deployment for benchmarking scifact-specific model.
Run: modal deploy modal_app_scifact.py
"""
import modal
import logging

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

app = modal.App("ranksaga-beir-benchmark-scifact", image=image)

# Volumes for persistent storage
results_volume = modal.Volume.from_name("ranksaga-benchmark-results", create_if_missing=True)
models_volume = modal.Volume.from_name("ranksaga-models", create_if_missing=True)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,  # 1 hour timeout
    volumes={"/results": results_volume, "/models": models_volume},
    memory=32768,
)
def benchmark_scifact_model(
    model_name: str = "ranksaga-optimized-e5-v2-scifact",
):
    """
    Benchmark the scifact-specific model on scifact dataset.
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
    
    logger.info(f"Starting benchmark for scifact-specific model: {model_name}")
    
    results_dir = Path("/results/scifact_specific")
    results_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("/datasets", exist_ok=True)
    
    if torch.cuda.is_available():
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ Using CPU")
    
    model_path = f"/models/{model_name}"
    logger.info(f"Loading model from: {model_path}")
    try:
        model = SentenceTransformer(model_path)
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception as e:
        logger.error(f"Error loading model {model_name} from {model_path}: {e}")
        return {"status": "error", "message": f"Failed to load model: {e}"}
    
    EVALUATION_METRICS = [1, 3, 5, 10, 100, 1000]
    
    def manual_retrieve(model, corpus, queries, top_k=1000):
        doc_texts = [corpus[doc_id]["text"] for doc_id in corpus.keys()]
        doc_ids = list(corpus.keys())
        batch_size = 256 if torch.cuda.is_available() else 32
        doc_embeddings = model.encode(doc_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, device="cuda" if torch.cuda.is_available() else "cpu")
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        query_texts = list(queries.values())
        query_ids = list(queries.keys())
        query_embeddings = model.encode(query_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, device="cuda" if torch.cuda.is_available() else "cpu")
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        scores = np.dot(doc_embeddings, query_embeddings.T)
        
        results_dict = {}
        for i, query_id in enumerate(tqdm(query_ids, desc="Building results")):
            top_indices = np.argsort(scores[:, i])[::-1][:top_k]
            results_dict[query_id] = {doc_ids[idx]: float(scores[idx, i]) for idx in top_indices}
        return results_dict
    
    def load_beir_dataset(dataset_name, datasets_dir, split="test"):
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
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name} on scifact dataset")
    logger.info(f"{'='*60}")
    
    try:
        corpus, queries, qrels = load_beir_dataset("scifact", "/datasets", split="test")
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
        
        logger.info(
            f"scifact Results:\n"
            f"  NDCG@10: {dataset_results['NDCG@10']:.4f}\n"
            f"  NDCG@100: {dataset_results['NDCG@100']:.4f}\n"
            f"  MAP@100: {dataset_results['MAP@100']:.4f}\n"
            f"  Recall@100: {dataset_results['Recall@100']:.4f}"
        )
        
        results_file = results_dir / f"{model_name.replace('-', '_')}_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "results": {"scifact": dataset_results}
            }, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {results_file}")
        results_volume.commit()
        logger.info("✅ Volume committed - results are persisted")
        
        return {
            "status": "success",
            "model": model_name,
            "results": {"scifact": dataset_results},
            "results_file": str(results_file)
        }
        
    except Exception as e:
        logger.error(f"Error evaluating scifact: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def main(model_name: str = "ranksaga-optimized-e5-v2-scifact"):
    """Run scifact benchmark via CLI: modal run modal_app_scifact.py"""
    import json
    result = benchmark_scifact_model.remote(model_name=model_name)
    print("\n" + "="*60)
    print("SciFact Benchmark Results")
    print("="*60)
    print(json.dumps(result, indent=2))
    return result

