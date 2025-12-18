import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import config
from dataset import get_data_loaders
from utils import plot_confusion_matrix

def main():
    # 1. Data Setup
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        config.CONFIG['data_dir'], 
        config.CONFIG['batch_size'], 
        config.CONFIG['image_size']
    )
    
    if not train_loader: 
        print("❌ Error: No data found.")
        return

    # 2. Model Setup (Swin Transformer)
    print("\n🏗️ Swin Transformer (PyTorch Native)...")
    model = models.swin_t(weights='DEFAULT')
    # Replace Head
    model.head = nn.Linear(model.head.in_features, len(class_names))
    model = model.to(config.CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=config.CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    print("🚀 Starting training...")
    for epoch in range(config.CONFIG['epochs']):
        print(f"Epoch {epoch+1}/{config.CONFIG['epochs']}")
        for phase in ['train', 'val']:
            if phase == 'train': 
                model.train()
                dl = train_loader
            else: 
                model.eval()
                dl = val_loader
            
            running_corrects = 0
            total = 0
            
            for inputs, labels in dl:
                inputs = inputs.to(config.CONFIG['device'])
                labels = labels.to(config.CONFIG['device'])
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)
            
            epoch_acc = running_corrects.double() / total
            print(f"   {phase.upper()} Acc: {epoch_acc:.4f}")
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

    # 4. Final Evaluation (Exact Metrics from Notebook)
    print("\n📝 --- CALCULATING METRICS (TEST SET) ---")
    model.load_state_dict(best_weights)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = [] 
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.CONFIG['device'])
            outputs = model(inputs)
            
            # Probs for AUC
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Metrics Calculation ---
    
    # 1. Class-wise Report
    class_report = classification_report(all_labels, all_preds, target_names=class_names, digits=2)

    # 2. Global Accuracy
    acc = accuracy_score(all_labels, all_preds)

    # 3. Macro Metrics
    prec_macro = precision_score(all_labels, all_preds, average='macro')
    rec_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # 4. Macro AUC
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"⚠️ Warning: AUC calculation issue: {e}")
        auc_macro = 0.0

    # --- Print Output ---
    print("\nClassification Report (Test Set): Swin Transformer")
    print(class_report)
    
    print("\n\n--- Test Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision (macro) : {prec_macro:.4f}")
    print(f"Recall (macro) : {rec_macro:.4f}")
    print(f"F1-score (macro) : {f1_macro:.4f}")
    print(f"AUC (macro) : {auc_macro:.12f}") 

    # Visualizations
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # Save
    print(f"\n💾 Saving: {config.CONFIG['save_path']}")
    torch.save(model.state_dict(), config.CONFIG['save_path'])
    print("✅ Done!")

if __name__ == '__main__':
    main()
