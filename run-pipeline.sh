#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the virtual environment
echo "Activating the virtual environment..."
source /home/administrateur/exo/mlops-simple/.venv/bin/activate

# Step 1: Prepare the data
echo "Preparing data..."
python3 /home/administrateur/exo/mlops-simple/scripts/prepare_data.py

# Step 2: Train the model
echo "Training the model..."
python3 /home/administrateur/exo/mlops-simple/scripts/train_model.py

# Step 3: Evaluate the model
echo "Evaluating the model..."
python3 /home/administrateur/exo/mlops-simple/scripts/evaluate_model.py

echo "Pipeline execution completed successfully."
