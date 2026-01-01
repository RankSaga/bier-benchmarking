"""
Export model for Hugging Face upload.
Downloads model from Modal if needed and prepares it for Hugging Face.
"""
import logging
import subprocess
import sys
from pathlib import Path
import shutil

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MODEL_NAME = "ranksaga-optimized-e5-v2"
MODAL_VOLUME = "ranksaga-models"
MODAL_MODEL_PATH = f"/models/{MODEL_NAME}"
LOCAL_MODEL_DIR = Path(__file__).parent / "models" / MODEL_NAME
EXPORT_DIR = Path(__file__).parent / "hf_export" / MODEL_NAME


def check_model_local() -> bool:
    """Check if model exists locally."""
    if LOCAL_MODEL_DIR.exists() and list(LOCAL_MODEL_DIR.glob("*.bin")):
        logger.info(f"✅ Model found locally at {LOCAL_MODEL_DIR}")
        return True
    logger.warning(f"⚠️  Model not found locally at {LOCAL_MODEL_DIR}")
    return False


def download_from_modal() -> bool:
    """Download model from Modal volume."""
    logger.info(f"Attempting to download model from Modal volume: {MODAL_VOLUME}")
    logger.info(f"Model path on Modal: {MODAL_MODEL_PATH}")
    
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to download the entire model directory
        cmd = [
            "modal", "volume", "download",
            MODAL_VOLUME,
            MODAL_MODEL_PATH,
            str(LOCAL_MODEL_DIR)
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Successfully downloaded model from Modal")
            return True
        else:
            logger.error(f"❌ Failed to download from Modal: {result.stderr}")
            logger.info("\n💡 Alternative: Download manually using:")
            logger.info(f"   modal volume download {MODAL_VOLUME} {MODAL_MODEL_PATH} {LOCAL_MODEL_DIR}")
            return False
    except FileNotFoundError:
        logger.error("❌ Modal CLI not found. Please install: pip install modal")
        logger.info("\n💡 Alternative: Download manually using Modal dashboard:")
        logger.info("   1. Go to https://modal.com")
        logger.info(f"   2. Navigate to volumes: {MODAL_VOLUME}")
        logger.info(f"   3. Download files from: {MODAL_MODEL_PATH}")
        return False


def prepare_for_huggingface() -> bool:
    """Prepare model files for Hugging Face upload."""
    if not LOCAL_MODEL_DIR.exists():
        logger.error(f"❌ Model directory not found: {LOCAL_MODEL_DIR}")
        return False
    
    # Check for required files
    required_files = ["config.json", "pytorch_model.bin"]
    missing_files = []
    
    for file in required_files:
        if not (LOCAL_MODEL_DIR / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"⚠️  Missing files: {missing_files}")
        logger.info("Model may need to be saved in SentenceTransformer format first")
        logger.info("The model should have been saved using SentenceTransformer.save()")
        return False
    
    # Create export directory
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    logger.info(f"Copying model files to {EXPORT_DIR}")
    
    # Copy all files from model directory
    for file in LOCAL_MODEL_DIR.rglob("*"):
        if file.is_file():
            relative_path = file.relative_to(LOCAL_MODEL_DIR)
            dest_path = EXPORT_DIR / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest_path)
            logger.info(f"  Copied: {relative_path}")
    
    logger.info(f"✅ Model prepared for Hugging Face at: {EXPORT_DIR}")
    return True


def main():
    """Main function."""
    logger.info("="*60)
    logger.info("Preparing model for Hugging Face upload")
    logger.info("="*60)
    
    # Check if model exists locally
    if not check_model_local():
        logger.info("\n" + "="*60)
        logger.info("Model not found locally. Attempting to download from Modal...")
        logger.info("="*60)
        
        if not download_from_modal():
            logger.error("\n❌ Could not download model. Please ensure:")
            logger.error("   1. Modal CLI is installed: pip install modal")
            logger.error("   2. You're authenticated: modal token new")
            logger.error("   3. Model exists on Modal volume")
            logger.error("\nAlternatively, ensure the model is available at:")
            logger.error(f"   {LOCAL_MODEL_DIR}")
            sys.exit(1)
    
    # Prepare for Hugging Face
    logger.info("\n" + "="*60)
    logger.info("Preparing model for Hugging Face...")
    logger.info("="*60)
    
    if not prepare_for_huggingface():
        logger.error("\n❌ Failed to prepare model for Hugging Face")
        logger.error("Please ensure the model was saved properly")
        sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Model preparation complete!")
    logger.info("="*60)
    logger.info(f"\nModel ready for upload at: {EXPORT_DIR}")
    logger.info("\nNext steps:")
    logger.info("  1. Review the model files")
    logger.info("  2. Create/update model card (README.md)")
    logger.info("  3. Run upload script: python upload_to_hf.py")


if __name__ == "__main__":
    main()

