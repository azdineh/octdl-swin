import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_labels, preds, class_names):
    """Plots and shows the confusion matrix."""
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    # Saves the plot to a file instead of just showing it, which is better for remote servers
    plt.savefig('confusion_matrix_swin.png')
    print("Confusion matrix saved as 'confusion_matrix_swin.png'")
    # plt.show() # Uncomment if running locally with a display
