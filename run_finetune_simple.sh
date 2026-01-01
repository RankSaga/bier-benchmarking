#!/bin/bash
# Simple fine-tuning runner with timeout

echo "Starting fine-tuning on Modal..."
echo "This may take 30-120 minutes"
echo ""
echo "Monitor progress at: https://modal.com/apps -> ranksaga-fine-tune"
echo ""

# Run with timeout to prevent hanging
timeout 300 modal run modal_fine_tune.py --epochs 1 --batch-size 16 2>&1 | head -100

echo ""
echo "If the command timed out, check the Modal dashboard for actual progress."
echo "The fine-tuning continues running on Modal even if this command exits."

