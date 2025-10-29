#!/bin/bash

# ===============================
# Project Automation Script
# ===============================
# This script syncs files and folders from a designated source folder
# to multiple project destinations.
#
# It performs two main functions:
# 1. Copies .csv and .json config files.
# 2. Runs the python script to convert source .png images to .npy
#    files in all destination directories.
#
# WARNING: Existing image folders will be permanently deleted.

# --- Configuration ---
SOURCE_DIR="./source_data/experiment_5"   # Change this to your meaningful source folder
SOURCE_IMAGE_DIR="$SOURCE_DIR/images"     # Location of source .png files

# --- Job 1: Overwrite transformation_details.csv ---
echo "Job 1: Overwriting transformation_details.csv..."
cp -f "$SOURCE_DIR/transformation_details.csv" ./transformation_details.csv
cp -f "$SOURCE_DIR/transformation_details.csv" ./code/data/transformation_details.csv
cp -f "$SOURCE_DIR/transformation_details.csv" ./code/data/images/transformation_details.csv
cp -f "$SOURCE_DIR/transformation_details.csv" ./code/images/transformation_details.csv
cp -f "$SOURCE_DIR/transformation_details.csv" ./data/transformation_details.csv
echo "Job 1 Complete."
echo "---------------------------------"

# --- Job 2: Convert and deploy .npy images ---
echo "Job 2: Deleting old image folders and running .npy conversion..."

# Define the destination directories for the .npy files
DEST_DIRS=(
    "./code/data/images"
    "./code/images"
    "./data/images"
    "./images"
)

# Check if conversion script exists
CONVERSION_SCRIPT="./convert_png_to_npy.py"
if [ ! -f "$CONVERSION_SCRIPT" ]; then
    echo "ERROR: Conversion script not found at $CONVERSION_SCRIPT"
    echo "Please ensure 'convert_png_to_npy.py' is in the project root."
    exit 1
fi

for dest_dir in "${DEST_DIRS[@]}"; do
    echo "  -> Removing old $dest_dir"
    rm -rf "$dest_dir"
    
    echo "  -> Running conversion: $SOURCE_IMAGE_DIR -> $dest_dir"
    # Run the Python script to convert PNGs from source to .npy in destination
    # We assume Python, numpy, and pillow are installed
    python "$CONVERSION_SCRIPT" --input_dir "$SOURCE_IMAGE_DIR" --output_dir "$dest_dir"
done

echo "Job 2 Complete."
echo "---------------------------------"

# --- Job 3: Overwrite example_config.json ---
echo "Job 3: Overwriting example_config.json..."
cp -f "$SOURCE_DIR/example_config.json" ./code/configs/example_config.json
echo "Job 3 Complete."
echo "---------------------------------"

# --- Job 4: Overwrite example_eval_config.json ---
echo "Job 4: Overwriting example_eval_config.json..."
cp -f "$SOURCE_DIR/example_eval_config.json" ./code/configs/example_eval_config.json
echo "Job 4 Complete."
echo "---------------------------------"

echo "All jobs completed successfully."
