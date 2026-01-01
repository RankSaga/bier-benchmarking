"""
Benchmark fine-tuned RankSaga-optimized models on BEIR datasets.
"""
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from beir.retrieval.evaluation import EvaluateRetrieval
import numpy as np

from config import (
    BEIR_DATASETS,
    PATHS,
    EVALUATION_METRICS,
    OPTIMIZED_MODEL_NAME
)
from utils.data_loader import load_beir_dataset
from utils.evaluation import format_metrics, save_results

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def manual_retrieve(model, corpus, queries, top_k=1000):
    """
    Manually perform dense retrieval using the model.
    """
    import numpy as np
    from tqdm import tqdm
    
    # Encode all documents
    logger.info("Encoding documents...")
    doc_texts = [corpus[doc_id]["text"] for doc_id in corpus.keys()]
    doc_ids = list(corpus.keys())
    doc_embeddings = model.encode(doc_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize embeddings for cosine similarity
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Encode queries and retrieve
    logger.info("Encoding queries and retrieving...")
    results_dict = {}
    
    for query_id, query_text in tqdm(queries.items(), desc="Processing queries"):
        # Encode query
        query_emb = model.encode([query_text], convert_to_numpy=True)[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Compute cosine similarity
        scores = np.dot(doc_embeddings, query_emb)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results dictionary
        results_dict[query_id] = {
            doc_ids[idx]: float(scores[idx]) for idx in top_indices
        }
    
    return results_dict


def benchmark_optimized_model(model_path: str = None):
    """
    Benchmark a fine-tuned optimized model on BEIR datasets.
    
    Args:
        model_path: Path to the fine-tuned model (defaults to config)
    """
    if model_path is None:
        model_path = str(PATHS["models"] / OPTIMIZED_MODEL_NAME)
    
    logger.info(f"Loading optimized model from: {model_path}")
    
    try:
        # Load fine-tuned model directly
        model = SentenceTransformer(model_path)
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        logger.info("Make sure you have run fine_tune_model.py first")
        return None
    
    all_results = {}
    
    for dataset_name in BEIR_DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating optimized model on dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Load dataset
            corpus, queries, qrels = load_beir_dataset(
                dataset_name,
                PATHS["datasets"],
                split="test"
            )
            
            # Retrieve manually
            logger.info("Running retrieval...")
            results_dict = manual_retrieve(model, corpus, queries, top_k=max(EVALUATION_METRICS))
            
            # Evaluate using BEIR's evaluator
            logger.info("Evaluating results...")
            evaluator = EvaluateRetrieval(k_values=EVALUATION_METRICS)
            ndcg, _map, recall, precision = evaluator.evaluate(
                qrels, results_dict, EVALUATION_METRICS
            )
            
            # Format and store results
            dataset_results = format_metrics(ndcg, _map, recall, precision)
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
            all_results[dataset_name] = {"error": str(e)}
            continue
    
    # Save results
    save_results(
        all_results,
        PATHS["optimized_results"],
        OPTIMIZED_MODEL_NAME.replace("-", "_")
    )
    
    logger.info("\n" + "="*60)
    logger.info("Optimized model benchmarking complete!")
    logger.info("="*60)
    
    return all_results


if __name__ == "__main__":
    benchmark_optimized_model()

