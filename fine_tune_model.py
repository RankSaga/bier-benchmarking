"""
Fine-tune embedding model with RankSaga optimization techniques.
"""
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses

from config import (
    BASE_MODEL,
    BEIR_DATASETS,
    TRAINING_CONFIG,
    PATHS
)
from utils.data_loader import prepare_training_data

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def fine_tune_model(
    base_model_name: str = None,
    output_name: str = None,
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None
):
    """
    Fine-tune an embedding model with RankSaga optimizations.
    
    Args:
        base_model_name: Base model to fine-tune (defaults to config)
        output_name: Output model name (defaults to config)
        num_epochs: Number of training epochs (defaults to config)
        batch_size: Training batch size (defaults to config)
        learning_rate: Learning rate (defaults to config)
    """
    # Use config defaults if not provided
    base_model_name = base_model_name or BASE_MODEL
    output_name = output_name or TRAINING_CONFIG["output_path"]
    num_epochs = num_epochs or TRAINING_CONFIG["epochs"]
    batch_size = batch_size or TRAINING_CONFIG["batch_size"]
    learning_rate = learning_rate or TRAINING_CONFIG["learning_rate"]
    
    logger.info(f"Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)
    
    # Prepare training data from multiple BEIR datasets
    logger.info("Preparing training data from BEIR datasets...")
    logger.info(f"Using datasets: {BEIR_DATASETS}")
    
    train_examples = prepare_training_data(BEIR_DATASETS, PATHS["datasets"])
    
    if len(train_examples) == 0:
        logger.error("No training examples prepared. Exiting.")
        return None
    
    # Create data loader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Use Multiple Negatives Ranking Loss for retrieval optimization
    logger.info("Setting up training loss: MultipleNegativesRankingLoss")
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Fine-tune the model
    logger.info(f"Starting fine-tuning for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        output_path=output_name,
        show_progress_bar=True,
        optimizer_params={'lr': learning_rate}
    )
    
    logger.info(f"Model fine-tuning complete!")
    logger.info(f"Model saved to: {output_name}")
    
    return model


if __name__ == "__main__":
    fine_tune_model()

