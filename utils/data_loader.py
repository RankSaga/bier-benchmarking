"""
BEIR dataset loading utilities.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from beir import util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)


def download_beir_dataset(dataset_name: str, datasets_dir: Path) -> Path:
    """
    Download a BEIR dataset if not already present.
    
    Args:
        dataset_name: Name of the BEIR dataset
        datasets_dir: Directory to store datasets
        
    Returns:
        Path to the downloaded dataset
    """
    dataset_path = datasets_dir / dataset_name
    
    if dataset_path.exists():
        logger.info(f"Dataset {dataset_name} already exists at {dataset_path}")
        return dataset_path
    
    logger.info(f"Downloading dataset {dataset_name}...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    
    try:
        data_path = util.download_and_unzip(url, str(datasets_dir))
        logger.info(f"Dataset {dataset_name} downloaded successfully")
        return Path(data_path)
    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_name}: {e}")
        raise


def load_beir_dataset(
    dataset_name: str,
    datasets_dir: Path,
    split: str = "test"
) -> Tuple[Dict, Dict, Dict]:
    """
    Load a BEIR dataset.
    
    Args:
        dataset_name: Name of the BEIR dataset
        datasets_dir: Directory containing datasets
        split: Dataset split to load ('train', 'test', 'dev')
        
    Returns:
        Tuple of (corpus, queries, qrels)
    """
    dataset_path = download_beir_dataset(dataset_name, datasets_dir)
    
    logger.info(f"Loading {dataset_name} {split} split...")
    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split=split)
    
    logger.info(
        f"Loaded {dataset_name}: {len(corpus)} documents, "
        f"{len(queries)} queries, {len(qrels)} query-document pairs"
    )
    
    return corpus, queries, qrels


def prepare_training_data(
    dataset_names: List[str],
    datasets_dir: Path
) -> List:
    """
    Prepare training data from multiple BEIR datasets.
    
    Args:
        dataset_names: List of BEIR dataset names
        datasets_dir: Directory containing datasets
        
    Returns:
        List of InputExample objects for training
    """
    from sentence_transformers import InputExample
    
    train_examples = []
    
    for dataset_name in dataset_names:
        try:
            corpus, queries, qrels = load_beir_dataset(
                dataset_name, datasets_dir, split="train"
            )
            
            # Create training pairs (query, positive passage)
            for query_id, query_text in queries.items():
                if query_id in qrels:
                    positive_passages = qrels[query_id]
                    for passage_id, score in positive_passages.items():
                        if score > 0 and passage_id in corpus:
                            passage_text = corpus[passage_id]["text"]
                            train_examples.append(
                                InputExample(
                                    texts=[query_text, passage_text],
                                    label=float(score)
                                )
                            )
            
            logger.info(
                f"Added {len([ex for ex in train_examples if dataset_name in str(ex)])} "
                f"training examples from {dataset_name}"
            )
        except Exception as e:
            logger.warning(
                f"Could not load training data from {dataset_name}: {e}. "
                f"Skipping this dataset."
            )
            continue
    
    logger.info(f"Total training examples prepared: {len(train_examples)}")
    return train_examples

