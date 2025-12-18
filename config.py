import os
import torch

# Paths
# Update this path to your local dataset location
INPUT_ROOT = '/kaggle/input/octdl-retinal-oct-images-dataset/OCTDL Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods/OCTDL/OCTDL' 
DATA_DIR = INPUT_ROOT if os.path.exists(INPUT_ROOT) else './OCTDL_dataset'

# Hyperparameters
CONFIG = {
    'data_dir': DATA_DIR, 
    'image_size': 224,
    'batch_size': 32,        
    'epochs': 20,            
    'learning_rate': 1e-4,   
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': 'swin_transformer_octdl_final.pth',
    'seed': 42
}

print(f"⚙️ Configuration loaded. Device: {CONFIG['device']}")
