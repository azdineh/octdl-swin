import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import config

class OCTDataset(Dataset):
    def __init__(self, dataframe, transform, mapping):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.mapping = mapping
        
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        label = self.mapping[row['label']]
        return self.transform(img), label

def get_data_loaders(data_dir, batch_size, img_size):
    image_paths = []
    labels = []
    
    # 1. Gather files
    for path in Path(data_dir).rglob('*'):
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_paths.append(str(path))
            labels.append(path.parent.name)
    
    df = pd.DataFrame({'filepath': image_paths, 'label': labels})
    
    if len(df) == 0: 
        return None, None, None, None
    
    # 2. Filter classes with <= 10 samples
    valid_counts = df['label'].value_counts()
    keep_labels = valid_counts[valid_counts > 10].index 
    df = df[df['label'].isin(keep_labels)]

    classes = sorted(df['label'].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # 3. Stratified Split (preserving exact logic from notebook)
    # 20% Test, then 15% of remaining for Val
    # Note: original code used 0.28 for the first split to achieve approx distribution desired
    train_val_df, test_df = train_test_split(
        df, test_size=0.28, stratify=df['label'], random_state=config.CONFIG['seed']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15, stratify=train_val_df['label'], random_state=config.CONFIG['seed']
    )

    # 4. Transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        norm
    ])
    
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm
    ])

    # 5. Loaders
    train_loader = DataLoader(
        OCTDataset(train_df, train_tf, class_to_idx), 
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        OCTDataset(val_df, eval_tf, class_to_idx), 
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        OCTDataset(test_df, eval_tf, class_to_idx), 
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, classes
