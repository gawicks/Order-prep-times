#!/bin/bash

echo "Setting up prep time prediction environment with Keras/TensorFlow..."

# Check if pip is available
if ! command -v pip &> /dev/null
then
    echo "pip could not be found. Please install pip first."
    exit
fi

# Install required packages
echo "Installing required packages..."
pip install tensorflow scikit-learn pandas numpy matplotlib

echo "Setup complete! You can now run:"
echo "python prep_time_keras_model.py"