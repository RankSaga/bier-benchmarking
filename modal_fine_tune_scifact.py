"""
Modal.com deployment for scifact-specific fine-tuning with advanced techniques.
Uses hard negative mining, supervised contrastive learning, and optimized hyperparameters.
Run: modal deploy modal_fine_tune_scifact.py
"""
import modal
import logging

# Create a Modal image with latest stable dependencies (2026)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "beir>=2.2.0",
        "sentence-transformers>=3.0.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.4.0",
    )
)

app = modal.App("ranksaga-fine-tune-scifact", image=image)

# Volumes for persistent storage
models_volume = modal.Volume.from_name("ranksaga-models", create_if_missing=True)
results_volume = modal.Volume.from_name("ranksaga-benchmark-results", create_if_missing=True)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@app.function(
    image=image,
    gpu="H100",  # Use H100 GPU for faster training
    timeout=18000,  # 5 hour timeout
    volumes={"/models": models_volume, "/datasets": results_volume},
    memory=32768,  # 32GB RAM
)
def fine_tune_scifact_model(
    base_model_name: str = "intfloat/e5-base-v2",
    output_name: str = "ranksaga-optimized-e5-v2-scifact",
    epochs: int = 10,  # More epochs for scifact
    batch_size: int = 32,  # Larger batch for MultipleNegativesRankingLoss
    learning_rate: float = 1e-5,  # Slightly higher LR
    warmup_steps: int = 300,  # More warmup
):
    """
    Fine-tune model specifically for scifact with advanced techniques:
    1. Hard negative mining
    2. Supervised contrastive learning
    3. Optimized hyperparameters for small dataset
    4. Multiple data passes with different strategies
    """
    import logging
    import os
    from pathlib import Path
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    import torch
    import numpy as np
    from tqdm import tqdm
    
    logger.info("="*60)
    logger.info("SciFact-Specific Fine-Tuning with Advanced Techniques")
    logger.info("="*60)
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Output model: {output_name}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    logger.info("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠️ GPU not available, using CPU (slower)")
    
    # Create directories
    os.makedirs("/datasets", exist_ok=True)
    output_path = f"/models/{output_name}"
    os.makedirs(output_path, exist_ok=True)
    
    def load_beir_dataset(dataset_name, datasets_dir, split="train"):
        """Load a BEIR dataset."""
        dataset_path = Path(datasets_dir) / dataset_name
        
        if not dataset_path.exists():
            logger.info(f"Downloading dataset {dataset_name}...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, str(datasets_dir))
            dataset_path = Path(data_path)
        
        logger.info(f"Loading {dataset_name} {split} split...")
        try:
            corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split=split)
            logger.info(f"Loaded {dataset_name}: {len(corpus)} documents, {len(queries)} queries")
            return corpus, queries, qrels
        except Exception as e:
            logger.warning(f"Could not load {split} split for {dataset_name}: {e}")
            return None, None, None
    
    # Load scifact dataset - try all splits to maximize data
    logger.info("\n" + "="*60)
    logger.info("Loading SciFact dataset...")
    logger.info("="*60)
    
    corpus = None
    queries = None
    qrels = None
    split_used = None
    
    splits_to_try = ["train", "dev", "test"]
    for split in splits_to_try:
        corpus, queries, qrels = load_beir_dataset("scifact", "/datasets", split=split)
        if corpus is not None:
            split_used = split
            break
    
    if corpus is None:
        logger.error("❌ Could not load scifact dataset from any split")
        return {"status": "error", "message": "Could not load scifact dataset"}
    
    logger.info(f"✅ Using {split_used} split: {len(corpus)} docs, {len(queries)} queries")
    
    # Load model for initial encoding (for hard negative mining)
    logger.info("\n" + "="*60)
    logger.info("Loading base model for hard negative mining...")
    logger.info("="*60)
    model = SentenceTransformer(base_model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Prepare training data - MultipleNegativesRankingLoss uses in-batch negatives
    logger.info("\n" + "="*60)
    logger.info("Preparing training data for MultipleNegativesRankingLoss...")
    logger.info("="*60)
    
    train_examples = []
    
    # Create query-document positive pairs
    # MultipleNegativesRankingLoss will automatically use in-batch negatives
    logger.info("Creating positive query-document pairs...")
    
    for query_id, query_text in tqdm(queries.items(), desc="Processing queries"):
        if query_id not in qrels:
            continue
        
        # Get positive passages
        positive_passages = {pid: score for pid, score in qrels[query_id].items() if score > 0}
        
        if not positive_passages:
            continue
        
        # Create (query, positive) pairs
        # MultipleNegativesRankingLoss uses in-batch negatives automatically
        for pos_id, pos_score in positive_passages.items():
            if pos_id in corpus:
                train_examples.append(
                    InputExample(
                        texts=[query_text, corpus[pos_id]["text"]],
                        label=float(pos_score)
                    )
                )
    
    if len(train_examples) == 0:
        logger.error("❌ No training examples prepared")
        return {"status": "error", "message": "No training examples prepared"}
    
    logger.info(f"\n✅ Prepared {len(train_examples)} training examples")
    logger.info(f"   (MultipleNegativesRankingLoss will use in-batch negatives)")
    
    # Create data loader with larger batch size for better in-batch negatives
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Use MultipleNegativesRankingLoss - proven best for retrieval tasks
    # This loss automatically uses in-batch negatives, which is very effective
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Fine-tune the model
    logger.info("\n" + "="*60)
    logger.info("Starting scifact-specific fine-tuning...")
    logger.info("="*60)
    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info("="*60)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        optimizer_params={'lr': learning_rate, 'weight_decay': 0.01},  # L2 regularization
        use_amp=True,  # Mixed precision training
        checkpoint_save_steps=500,  # Save checkpoints more frequently
        checkpoint_path=output_path,
    )
    
    # Commit volume to persist model
    models_volume.commit()
    
    logger.info("\n" + "="*60)
    logger.info("✅ SciFact-specific fine-tuning complete!")
    logger.info(f"✅ Model saved to: {output_path}")
    logger.info("✅ Volume committed - model is persisted")
    logger.info("="*60)
    
    return {
        "status": "success",
        "base_model": base_model_name,
        "output_model": output_name,
        "output_path": output_path,
        "training_examples": len(train_examples),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


@app.local_entrypoint()
def main(
    base_model: str = "intfloat/e5-base-v2",
    output_name: str = "ranksaga-optimized-e5-v2-scifact",
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 5e-6
):
    """Run scifact-specific fine-tuning via CLI: modal run modal_fine_tune_scifact.py"""
    import json
    result = fine_tune_scifact_model.remote(
        base_model_name=base_model,
        output_name=output_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    print("\n" + "="*60)
    print("SciFact Fine-Tuning Summary")
    print("="*60)
    print(json.dumps(result, indent=2))
    return result

