"""
Modal.com deployment for fine-tuning embedding models.
Run: modal deploy modal_fine_tune.py
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
        "accelerate>=0.26.0",  # Required for Trainer with PyTorch
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
    )
)

app = modal.App("ranksaga-fine-tune", image=image)

# Volume for persistent storage of models
models_volume = modal.Volume.from_name("ranksaga-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # Use A10G GPU for fast training
    timeout=14400,  # 4 hour timeout for training
    volumes={"/models": models_volume},
    memory=32768,  # 32GB RAM
)
def fine_tune_model(
    base_model_name: str = "intfloat/e5-base-v2",
    output_name: str = "ranksaga-optimized-e5-v1",
    num_epochs: int = 3,
    batch_size: int = 32,  # Larger batch size for GPU
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    datasets: list = None
):
    """
    Fine-tune an embedding model with RankSaga optimizations on Modal.
    
    Args:
        base_model_name: Base model to fine-tune
        output_name: Output model name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        datasets: List of BEIR datasets to use for training
    """
    import logging
    import os
    from pathlib import Path
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    import torch
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    if datasets is None:
        # Only use datasets that have train splits
        # scidocs and quora may not have train splits, so we'll try them but skip if they fail
        datasets = ["scifact", "nfcorpus"]  # Start with datasets that definitely have train splits
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠️ GPU not available, using CPU (slower)")
    
    # Create directories
    os.makedirs("datasets", exist_ok=True)
    output_path = f"/models/{output_name}"
    os.makedirs(output_path, exist_ok=True)
    
    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Prepare training data
    logger.info("Preparing training data from BEIR datasets...")
    train_examples = []
    
    def load_beir_dataset(dataset_name, datasets_dir, split="train"):
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
    
    for dataset_name in datasets:
        try:
            corpus, queries, qrels = load_beir_dataset(dataset_name, "datasets", split="train")
            
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
            
            logger.info(f"Added training examples from {dataset_name}. Total: {len(train_examples)}")
        except Exception as e:
            logger.warning(f"Could not load training data from {dataset_name}: {e}. Skipping.")
            continue
    
    if len(train_examples) == 0:
        logger.error("No training examples prepared. Exiting.")
        return {"status": "error", "message": "No training examples prepared"}
    
    logger.info(f"Total training examples: {len(train_examples)}")
    
    # Create data loader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Use Multiple Negatives Ranking Loss
    logger.info("Setting up training loss: MultipleNegativesRankingLoss")
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Fine-tune the model
    logger.info(f"Starting fine-tuning for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        optimizer_params={'lr': learning_rate}
    )
    
    # Commit volume to persist model
    models_volume.commit()
    
    logger.info(f"✅ Model fine-tuning complete!")
    logger.info(f"✅ Model saved to: {output_path}")
    logger.info("✅ Volume committed - model is persisted")
    
    return {
        "status": "success",
        "base_model": base_model_name,
        "output_model": output_name,
        "output_path": output_path,
        "training_examples": len(train_examples),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }


@app.local_entrypoint()
def main(
    base_model: str = "intfloat/e5-base-v2",
    output_name: str = "ranksaga-optimized-e5-v1",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5
):
    """Run fine-tuning via CLI: modal run modal_fine_tune.py --epochs 3"""
    import json
    result = fine_tune_model.remote(
        base_model_name=base_model,
        output_name=output_name,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    print("\n" + "="*60)
    print("Fine-Tuning Summary")
    print("="*60)
    print(json.dumps(result, indent=2))
    return result

