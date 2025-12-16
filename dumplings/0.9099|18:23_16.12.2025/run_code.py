import os
import shutil
import subprocess
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from torchvision.models import EfficientNet_V2_S_Weights
from tqdm.auto import tqdm
from PIL import Image
from IPython import get_ipython
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             auc, precision_recall_curve, average_precision_score, accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter
import kagglehub
from google.colab import userdata
import gc
import inspect
import pprint
from google.colab import _message

# %% [CELL] 

token = userdata.get('GITHUB_DLTESTWORK_TOKEN') or os.environ.get('GITHUB_DLTESTWORK_TOKEN')
path_filtered = kagglehub.dataset_download("kvdep1/dumplings")
path_sub = kagglehub.competition_download("yummi-classification-fu25")

# %% [CELL] 

class GitLogger:
    def __init__(self, repo_url, token, subdir="dumplings"):
        self.repo_name = "DL-TESTWORK"
        self.root_subdir = subdir
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        self.auth_url = f"https://{token}@{clean_url}"
        self.base_path = os.path.abspath(self.repo_name)
        self.current_exp_folder = None
        self._setup_repo()

    def _run(self, cmd, cwd=None):
        if cwd is None: cwd = self.base_path
        print(f"[RUN] {cmd}")
        try:
            subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {cmd} failed: {e.stderr}")

    def _setup_repo(self):
        if os.path.exists(self.base_path): shutil.rmtree(self.base_path)
        print("[INFO] Cloning repository...")
        subprocess.check_call(f"git clone {self.auth_url} {self.repo_name}", shell=True)
        self._run("git config user.email 'bot@colab.com'")
        self._run("git config user.name 'ColabBot'")

    def start_experiment(self, val_acc):
        now = datetime.now()
        date_str = now.strftime("%H:%M_%d.%m.%Y")

        self.last_val_acc = val_acc
        folder_name = f"{val_acc:.4f}|{date_str}"
        self.current_exp_folder = os.path.join(self.base_path, self.root_subdir, folder_name)

        os.makedirs(self.current_exp_folder, exist_ok=True)
        print(f"[INFO] Created folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder: return
        target = os.path.join(self.current_exp_folder, "run_code.py")

        try:
            # 1. Get the actual notebook JSON from Colab backend
            nb = _message.blocking_request('get_ipynb')

            code_content = []

            # 2. Extract source from Code cells only
            if nb and 'ipynb' in nb and 'cells' in nb['ipynb']:
                for cell in nb['ipynb']['cells']:
                    if cell['cell_type'] == 'code':
                        # Join lines of code in the cell
                        source = "".join(cell['source'])
                        code_content.append(source)

            full_text = "\n\n# %% [CELL] \n\n".join(code_content)

            # 3. REDACT TOKEN (Mandatory for GitHub Push)
            # We must remove the token string from the text before saving
            try:
                token_str = self.auth_url.split('//')[1].split('@')[0]
                if len(token_str) > 5:
                    full_text = full_text.replace(token_str, "REDACTED_TOKEN")
            except:
                pass

            with open(target, 'w', encoding='utf-8') as f:
                f.write(full_text)

            print(f"[INFO] Notebook code (untouched) saved to {target}")

        except Exception as e:
            print(f"[ERROR] Failed to save notebook code: {e}")

    def save_figure(self, fig, filename):
        if not self.current_exp_folder: return
        path = os.path.join(self.current_exp_folder, filename)
        fig.savefig(path, bbox_inches='tight', dpi=120, facecolor=fig.get_facecolor())
        print(f"[INFO] Figure saved: {filename}")

    def save_csv(self, df, filename):
        if not self.current_exp_folder: return
        path = os.path.join(self.current_exp_folder, filename)
        df.to_csv(path, index=False)
        print(f"[INFO] CSV saved: {filename}")

    def push_results(self):
        print("[INFO] Pushing...")
        self._run("git pull origin main")
        self._run("git add .")

        msg = f"{self.last_val_acc:.4f}"
        self._run(f"git commit -m '{msg}'")
        self._run("git push origin main")
        print("[SUCCESS] Pushed.")

# %% [CELL] 

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

# %% [CELL] 

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

# %% [CELL] 

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
    #plt.style.use('dark_background')
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
    ax1.set_title('Training Dynamics', fontsize=14, fontweight='bold', color='black')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.grid(True, alpha=0.1)

    # B. Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax2,
                xticklabels=classes, yticklabels=classes, cbar=False)
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='black')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # C. Per-Class Metrics Heatmap (Prec, Recall, F1)
    ax3 = fig.add_subplot(gs[0, 2])
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_rep = pd.DataFrame(rep).T.iloc[:-3, :3] # Exclude avg/accuracy
    sns.heatmap(df_rep, annot=True, fmt='.3f', cmap='viridis', ax=ax3, cbar=False)
    ax3.set_title('Class-wise Metrics', fontsize=14, fontweight='bold', color='black')

    # D. ROC Curves
    ax4 = fig.add_subplot(gs[1, 0])
    y_bin = label_binarize(y_true, classes=range(n_classes))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC={roc_auc:.2f})')
    ax4.plot([0, 1], [0, 1], 'w--', lw=1)
    ax4.legend(loc='lower right', fontsize=9)
    ax4.set_title('ROC Curves', fontsize=14, fontweight='bold', color='black')

    # E. Precision-Recall Curves
    ax5 = fig.add_subplot(gs[1, 1])
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax5.plot(recall, precision, lw=2, label=f'{classes[i]} (AP={ap:.2f})')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.legend(loc='lower left', fontsize=9)
    ax5.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold', color='black')

    # F. Confidence Histogram (Correct vs Wrong)
    ax6 = fig.add_subplot(gs[1, 2])
    max_probs = np.max(y_prob, axis=1)
    correct_mask = (y_pred == y_true)

    ax6.hist(max_probs[correct_mask], bins=20, alpha=0.7, color='green', label='Correct')
    ax6.hist(max_probs[~correct_mask], bins=20, alpha=0.7, color='red', label='Wrong')
    ax6.set_title('Confidence Distribution', fontsize=14, fontweight='bold', color='black')
    ax6.legend()

    # G. Global Metrics Text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    acc = accuracy_score(y_true, y_pred)
    txt = f"Global Accuracy: {acc:.2%}\n"
    txt += f"Best Validation Loss: {min(history['val_acc']):.4f} (approx)\n"
    txt += f"Total Samples: {len(y_true)}"
    ax7.text(0.5, 0.5, txt, ha='center', va='center', fontsize=20, color='black')

    # Save
    logger.save_figure(fig, "comprehensive_report.png")
    plt.close(fig)

    # Save CSV
    logger.save_csv(pd.DataFrame(rep).T, "detailed_metrics.csv")

# %% [CELL] 

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = list(Path(root).glob("*.*"))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path.name

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

# %% [CELL] 

def free_gpu():
    if 'model' in globals(): del globals()['model']
    if 'optimizer' in globals(): del globals()['optimizer']
    if 'scheduler' in globals(): del globals()['scheduler']
    gc.collect()
    torch.cuda.empty_cache()
free_gpu()

# %% [CELL] 

CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test/test/images',
    'batch_size': 32,
    'img_size' : 338,
    'epochs': 10,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

model, val_loader, class_names, history, best_acc = train()

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

logger.start_experiment(best_acc)
logger.save_code()

run_full_evaluation(model, val_loader, logger, history, class_names, CONFIG)

create_submission(model, class_names, logger, CONFIG)

logger.push_results()