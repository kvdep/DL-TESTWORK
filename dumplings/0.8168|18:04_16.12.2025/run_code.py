from IPython import get_ipython
from PIL import Image
from datetime import datetime
from google.colab import userdata
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from torchvision.models import EfficientNet_V2_S_Weights
from tqdm.auto import tqdm
import gc
import inspect
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import seaborn as sns
import shutil
import subprocess
import torch
import torch.nn as nn

# --- Definitions ---

CONFIG = {'batch_size': 16,
 'data_dir': '/content/datasets/kvdep1/dumplings/versions/5/dataset_filtered',
 'device': 'cuda',
 'epochs': 1,
 'img_size': 224,
 'lr': 0.001,
 'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
 'test_dir': '/content/competitions/yummi-classification-fu25/test/test/images',
 'token': 'REDACTED'}

def get_dataloaders():
    tf = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_ds = datasets.ImageFolder(CONFIG['data_dir'], transform=tf)
    train_sz = int(0.8 * len(full_ds))
    val_sz = len(full_ds) - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz])
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    return train_dl, val_dl, full_ds.classes


def build_model(num_classes):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(CONFIG['device'])


def train():
    train_loader, val_loader, classes = get_dataloaders()
    model = build_model(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['lr'], 
                                                    steps_per_epoch=len(train_loader), epochs=CONFIG['epochs'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    history = {'train_loss': [], 'val_acc': []}
    best_acc = 0.0

    print("Starting Training...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                out = model(x)
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        
        acc = correct / total
        history['val_acc'].append(acc)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            
    return model, val_loader, classes, history, best_acc


def create_submission(model, classes, logger, config):
    test_tf = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ds = TestDataset(config['test_dir'], test_tf)
    if len(ds) == 0:
        print("Skipping submission: No files found.")
        return

    loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False)
    results = []
    
    model.eval()
    print("Generating Submission...")
    with torch.no_grad():
        for x, fnames in loader:
            x = x.to(config['device'])
            _, preds = torch.max(model(x), 1)
            for f, p in zip(fnames, preds):
                results.append({'filename': f, 'class': classes[p.item()]})
    
    df = pd.DataFrame(results)
    logger.save_csv(df, "submission.csv")


def smooth(y, window=5, poly=2):
    if len(y) < window: return y
    return savgol_filter(y, window, poly)


def run_full_evaluation(model, loader, logger, history, classes, config):
    print("Running Full Evaluation...")
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    # 1. Inference
    with torch.no_grad():
        for x, y in loader:
            x = x.to(config['device'])
            out = model(x)
            prob = torch.nn.functional.softmax(out, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.max(prob, 1)[1].cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    n_classes = len(classes)
    
    # 2. Setup Visualization
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3)
    
    # A. Training Dynamics (Loss & Acc)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    epochs = range(1, len(history['train_loss']) + 1)
    l1 = ax1.plot(epochs, history['train_loss'], 'r-o', lw=2, label='Train Loss')
    l2 = ax1_twin.plot(epochs, history['val_acc'], 'c-o', lw=2, label='Val Acc')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='r')
    ax1_twin.set_ylabel('Accuracy', color='c')
    ax1.set_title('Training Dynamics', fontsize=14, fontweight='bold', color='white')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.grid(True, alpha=0.1)

    # B. Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax2, 
                xticklabels=classes, yticklabels=classes, cbar=False)
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # C. Per-Class Metrics Heatmap (Prec, Recall, F1)
    ax3 = fig.add_subplot(gs[0, 2])
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_rep = pd.DataFrame(rep).T.iloc[:-3, :3] # Exclude avg/accuracy
    sns.heatmap(df_rep, annot=True, fmt='.3f', cmap='viridis', ax=ax3, cbar=False)
    ax3.set_title('Class-wise Metrics', fontsize=14, fontweight='bold', color='white')

    # D. ROC Curves
    ax4 = fig.add_subplot(gs[1, 0])
    y_bin = label_binarize(y_true, classes=range(n_classes))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC={roc_auc:.2f})')
    ax4.plot([0, 1], [0, 1], 'w--', lw=1)
    ax4.legend(loc='lower right', fontsize=9)
    ax4.set_title('ROC Curves', fontsize=14, fontweight='bold', color='white')

    # E. Precision-Recall Curves
    ax5 = fig.add_subplot(gs[1, 1])
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax5.plot(recall, precision, lw=2, label=f'{classes[i]} (AP={ap:.2f})')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.legend(loc='lower left', fontsize=9)
    ax5.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold', color='white')

    # F. Confidence Histogram (Correct vs Wrong)
    ax6 = fig.add_subplot(gs[1, 2])
    max_probs = np.max(y_prob, axis=1)
    correct_mask = (y_pred == y_true)
    
    ax6.hist(max_probs[correct_mask], bins=20, alpha=0.7, color='green', label='Correct')
    ax6.hist(max_probs[~correct_mask], bins=20, alpha=0.7, color='red', label='Wrong')
    ax6.set_title('Confidence Distribution', fontsize=14, fontweight='bold', color='white')
    ax6.legend()

    # G. Global Metrics Text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    acc = accuracy_score(y_true, y_pred)
    txt = f"Global Accuracy: {acc:.2%}\n"
    txt += f"Best Validation Loss: {min(history['val_acc']):.4f} (approx)\n"
    txt += f"Total Samples: {len(y_true)}"
    ax7.text(0.5, 0.5, txt, ha='center', va='center', fontsize=20, color='white')

    # Save
    logger.save_figure(fig, "comprehensive_report.png")
    plt.close(fig)
    
    # Save CSV
    logger.save_csv(pd.DataFrame(rep).T, "detailed_metrics.csv")


def free_gpu():
    if 'model' in globals(): del globals()['model']
    if 'optimizer' in globals(): del globals()['optimizer']
    if 'scheduler' in globals(): del globals()['scheduler']
    gc.collect()
    torch.cuda.empty_cache()


def visualize_elegant(y_true, y_pred, classes, logger):
    # Set dark theme for vibrancy
    plt.style.use('dark_background')
    
    # Calculate Metrics
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    # Filter out accuracy/macro avg for the heatmap, keep classes
    class_metrics = {k: v for k, v in rep.items() if k in classes}
    df_metrics = pd.DataFrame(class_metrics).T.iloc[:, :3] # Precision, Recall, F1
    
    # Create Figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])

    # Plot A: Vibrant Heatmap
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="plasma", 
                linewidths=1, linecolor='black', cbar=False, ax=ax1,
                annot_kws={"size": 14, "weight": "bold"})
    ax1.set_title("Classification Quality", fontsize=18, color='white', pad=20)
    ax1.tick_params(axis='y', rotation=0, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    # Plot B: Accuracy Gauge (Donut Chart)
    ax2 = fig.add_subplot(gs[1])
    acc = rep['accuracy']
    
    # Donut data
    sizes = [acc, 1-acc]
    colors = ['#00ffcc', '#222222'] # Neon cyan and dark grey
    
    ax2.pie(sizes, labels=['', ''], colors=colors, startangle=90, 
            wedgeprops={'edgecolor': 'black', 'linewidth': 2, 'width': 0.3})
    
    # Central Text
    ax2.text(0, 0, f"{acc:.2%}", ha='center', va='center', fontsize=35, 
             fontweight='bold', color='white')
    ax2.text(0, -0.25, "Accuracy", ha='center', va='center', fontsize=14, color='gray')
    ax2.set_title("Model Performance", fontsize=18, color='white')

    # Save
    logger.save_figure(fig, "report_card.png")
    plt.close(fig)

