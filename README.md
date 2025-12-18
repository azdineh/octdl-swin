# OCTDL Classification with Swin Transformer

This repository contains the implementation for training a Swin Transformer (Tiny) on the OCTDL dataset. It includes specific metric calculations (AUC, Macro F1/Precision/Recall) and Stratified splitting.

## Project Structure

* `config.py`: Configuration for paths, hyperparameters, and device.
* `dataset.py`: Data loading, filtering (>10 samples), stratified splitting, and transformations.
* `train.py`: Main training loop and specific metric evaluation.
* `utils.py`: Helper functions for visualization.

## Requirements

* Python 3.8+
* PyTorch
* torchvision
* scikit-learn
* pandas
* matplotlib
* seaborn

Install dependencies via:
```bash
pip install -r requirements.txt
