#!/bin/bash

# Define directories
CACHE_DIR="/home/deepu/.cache/huggingface/hub/datasets--ai4bharat--sangraha"
PROJECT_DIR="/home/deepu/lmai/mp1"
TELUGU_DIR="$PROJECT_DIR/data/telugu/downloaded/raw_parquets"
NEPALI_DIR="$PROJECT_DIR/data/nepali/downloaded/raw_parquets"

# Create destination directories
mkdir -p "$TELUGU_DIR"
mkdir -p "$NEPALI_DIR"

echo "=== Moving Telugu Parquets ==="
# Find all telugu parquets in the snapshots folder. 
# We use 'cp -L' (dereference) because Hugging Face uses symlinks pointing to hidden blobs.
# Moving the symlink alone would break the file.
find "$CACHE_DIR/snapshots" -type l -path "*/verified/tel/*.parquet" 2>/dev/null | while read -r file; do
    filename=$(basename "$file")
    echo "Transferring $filename..."
    cp -L "$file" "$TELUGU_DIR/$filename"
done

echo ""
echo "=== Moving Nepali Parquets ==="
find "$CACHE_DIR/snapshots" -type l -path "*/verified/nep/*.parquet" 2>/dev/null | while read -r file; do
    filename=$(basename "$file")
    echo "Transferring $filename..."
    cp -L "$file" "$NEPALI_DIR/$filename"
done

echo ""
echo "==========================================="
echo "All Parquet files have been securely copied!"
echo "Destination Telugu: $TELUGU_DIR"
echo "Destination Nepali: $NEPALI_DIR"
echo "==========================================="
echo ""
echo "Note: The original blobs are still taking up space in your HuggingFace cache."
echo "If you want to clear that disk space, you can run:"
echo "rm -rf /home/deepu/.cache/huggingface/hub/datasets--ai4bharat--sangraha"
