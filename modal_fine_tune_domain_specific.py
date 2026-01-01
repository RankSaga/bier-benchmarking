"""
Domain-specific Modal.com deployment for fine-tuning embedding models.
Trains separate models for different domains to maximize performance.
Run: modal deploy modal_fine_tune_domain_specific.py
"""
import modal

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

app = modal.App("ranksaga-fine-tune-domain-specific", image=image)

# Volumes for persistent storage
models_volume = modal.Volume.from_name("ranksaga-models", create_if_missing=True)
results_volume = modal.Volume.from_name("ranksaga-benchmark-results", create_if_missing=True)

# Domain definitions
DOMAINS = {
    "scientific": {
        "datasets": ["scifact", "nfcorpus", "scidocs"],
        "output_name": "ranksaga-optimized-e5-v2-scientific",
        "description": "Scientific/Medical domain (scifact, nfcorpus, scidocs)"
    },
    "general": {
        "datasets": ["quora"],
        "output_name": "ranksaga-optimized-e5-v2-general",
        "description": "General Q&A domain (quora)"
    }
}


@app.function(
    image=image,
    gpu="H100",  # Use H100 GPU for faster training
    timeout=18000,  # 5 hour timeout
    volumes={"/models": models_volume, "/datasets": results_volume},
    memory=32768,  # 32GB RAM
)
def fine_tune_domain_model(
    domain: str = "scientific",
    base_model_name: str = "intfloat/e5-base-v2",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    warmup_steps: int = 500,
):
    """
    Fine-tune a domain-specific model.
    
    Args:
        domain: Domain to train for ("scientific" or "general")
        base_model_name: Base model to fine-tune
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps
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
    
    if domain not in DOMAINS:
        return {"status": "error", "message": f"Invalid domain: {domain}. Use 'scientific' or 'general'"}
    
    domain_config = DOMAINS[domain]
    datasets = domain_config["datasets"]
    output_name = domain_config["output_name"]
    
    logger.info("="*60)
    logger.info(f"Domain-Specific Fine-Tuning: {domain.upper()}")
    logger.info(f"Description: {domain_config['description']}")
    logger.info(f"Datasets: {datasets}")
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
    
    # Prepare training data from domain-specific datasets
    logger.info("="*60)
    logger.info("Preparing domain-specific training data...")
    logger.info("="*60)
    
    train_examples = []
    splits_to_try = ["train", "dev", "test"]
    
    for dataset_name in datasets:
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Try to load any available split
        corpus = None
        queries = None
        qrels = None
        split_used = None
        
        for split in splits_to_try:
            corpus, queries, qrels = load_beir_dataset(dataset_name, "/datasets", split=split)
            if corpus is not None:
                split_used = split
                break
        
        if corpus is None:
            logger.warning(f"⚠️ Could not load any split for {dataset_name}, skipping...")
            continue
        
        logger.info(f"✅ Using {split_used} split: {len(corpus)} docs, {len(queries)} queries")
        
        # Create training pairs (query, positive passage)
        examples_from_dataset = 0
        for query_id, query_text in queries.items():
            if query_id not in qrels:
                continue
            
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
                    examples_from_dataset += 1
        
        logger.info(f"✅ Added {examples_from_dataset} training pairs from {dataset_name}")
    
    if len(train_examples) == 0:
        logger.error("❌ No training examples prepared. Exiting.")
        return {"status": "error", "message": "No training examples prepared"}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Total training examples for {domain} domain: {len(train_examples)}")
    logger.info(f"{'='*60}\n")
    
    # Load model for fine-tuning
    logger.info(f"Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Create data loader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Use Multiple Negatives Ranking Loss
    logger.info("Setting up training loss: MultipleNegativesRankingLoss")
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Calculate total steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    logger.info(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    
    # Fine-tune the model
    logger.info("="*60)
    logger.info("Starting domain-specific fine-tuning...")
    logger.info(f"Domain: {domain}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info("="*60)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        optimizer_params={'lr': learning_rate},
        use_amp=True,  # Mixed precision
        checkpoint_path=f"{output_path}/checkpoints",
        checkpoint_save_steps=steps_per_epoch,
    )
    
    # Commit volume to persist model
    models_volume.commit()
    
    logger.info("\n" + "="*60)
    logger.info(f"✅ Domain-specific fine-tuning complete for {domain}!")
    logger.info(f"✅ Model saved to: {output_path}")
    logger.info("✅ Volume committed - model is persisted")
    logger.info("="*60)
    
    return {
        "status": "success",
        "domain": domain,
        "base_model": base_model_name,
        "output_model": output_name,
        "output_path": output_path,
        "training_examples": len(train_examples),
        "datasets_used": datasets,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


@app.function(
    image=image,
    gpu="H100",  # Use H100 GPU for faster training
    timeout=18000,
    volumes={"/models": models_volume, "/datasets": results_volume},
    memory=32768,
)
def fine_tune_all_domains(
    base_model_name: str = "intfloat/e5-base-v2",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    warmup_steps: int = 500,
):
    """
    Fine-tune models for all domains sequentially.
    """
    import logging
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    results = {}
    
    for domain in DOMAINS.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {domain} domain model...")
        logger.info(f"{'='*60}\n")
        
        result = fine_tune_domain_model.remote(
            domain=domain,
            base_model_name=base_model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        )
        
        results[domain] = result
    
    return results


@app.local_entrypoint()
def main(
    domain: str = "scientific",
    base_model: str = "intfloat/e5-base-v2",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    warmup_steps: int = 500,
    all_domains: bool = False,
):
    """
    Run domain-specific fine-tuning.
    
    Args:
        domain: Domain to train ("scientific" or "general")
        base_model: Base model name
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        all_domains: If True, train all domains sequentially
    """
    import json
    
    if all_domains:
        print("\n" + "="*60)
        print("Training ALL Domain-Specific Models")
        print("="*60)
        result = fine_tune_all_domains.remote(
            base_model_name=base_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        )
    else:
        print("\n" + "="*60)
        print(f"Domain-Specific Fine-Tuning: {domain.upper()}")
        print("="*60)
        result = fine_tune_domain_model.remote(
            domain=domain,
            base_model_name=base_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        )
    
    print("\n" + "="*60)
    print("Fine-Tuning Summary")
    print("="*60)
    print(json.dumps(result, indent=2))
    return result

