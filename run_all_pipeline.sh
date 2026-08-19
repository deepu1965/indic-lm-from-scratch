#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================="
echo " Starting Full Data & Tokenizer Pipeline "
echo "============================================="

# 1. Clean Telugu Data
echo ""
echo ">>> [1/3] Running Full Cleaning Pipeline for Telugu..."
python full_cleaning_pipeline.py --language telugu

# 2. Clean Nepali Data
echo ""
echo ">>> [2/3] Running Full Cleaning Pipeline for Nepali..."
python full_cleaning_pipeline.py --language nepali

# 3. Train Tokenizers
echo ""
echo ">>> [3/3] Training BPE Tokenizers (5k, 8k, 10k) for both languages..."
python train_tokenizers.py

echo ""
echo "============================================="
echo " 🎉 Pipeline Completed Successfully! 🎉 "
echo "============================================="
