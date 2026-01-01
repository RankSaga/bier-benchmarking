#!/bin/bash
# Quick status check script

echo "============================================================"
echo "RankSaga Benchmarking Status Check"
echo "============================================================"
echo ""

echo "📊 Modal Apps:"
modal app list 2>&1 | grep ranksaga | head -5

echo ""
echo "🔍 Fine-tuning app logs (last 20 lines):"
modal app logs ranksaga-fine-tune 2>&1 | tail -20

echo ""
echo "💡 To view full logs: modal app logs ranksaga-fine-tune"
echo "💡 Dashboard: https://modal.com/apps"

