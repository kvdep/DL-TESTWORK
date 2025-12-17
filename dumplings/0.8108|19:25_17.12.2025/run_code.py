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
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

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
    # Размер изображения из конфига
    img_size = CONFIG['img_size']

    # 1. Аугментации для ТРЕНИРОВКИ (Сильные)
    train_tf = transforms.Compose([
        # 1. Сразу чиним каналы
        transforms.Lambda(lambda x: x.convert('RGB')),

        # 2. Геометрия (чтобы модель узнавала объект боком, криво, косо)
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3), # Пельмени могут лежать как угодно
        transforms.RandomRotation(degrees=45), # Сильный поворот

        # Перспектива и искажения (очень полезно против фото под углом)
        transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15),

        # 3. Качество фото ("мусор", размытие, шумы)
        # Имитация плохой камеры / расфокуса
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ], p=0.3),

        # Сильное изменение цвета (яркость/тени)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

        # Иногда делаем ЧБ, чтобы модель учила форму, а не только цвет теста
        transforms.RandomGrayscale(p=0.1),

        # 4. Тензоры и нормализация
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        # 5. Вырезание кусков (Cutout) - заставляет восстанавливать скрытые части
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    # Валидация (стандартная)
    val_tf = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Загружаем датасеты с РАЗНЫМИ трансформациями
    # Важно: ImageFolder применяется ко всему датасету сразу, поэтому нам нужно разделить его хитро.
    # Самый простой способ в вашем коде (через random_split) имеет недостаток:
    # он не позволяет легко назначить разные transforms для train и val.
    # Поэтому загрузим данные дважды (это не дублирует файлы в памяти, только ссылки).

    full_data_train = datasets.ImageFolder(CONFIG['data_dir'], transform=train_tf)
    full_data_val   = datasets.ImageFolder(CONFIG['data_dir'], transform=val_tf)

    ds_len = len(full_data_train)
    train_sz = int(0.8 * ds_len)
    val_sz = ds_len - train_sz

    generator = torch.Generator().manual_seed(42)
    train_ds, _ = random_split(full_data_train, [train_sz, val_sz], generator=generator)
    _, val_ds   = random_split(full_data_val,   [train_sz, val_sz], generator=generator)

    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    return train_dl, val_dl, full_data_train.classes

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

    # --- Styling & Colors ---
    MAIN_RED = '#C62828'
    CLS_COLORS = ['#C62828', '#1565C0', '#F9A825', '#2E7D32']

    # Increase base font sizes significantly
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'font.size': 14,
        'axes.titlesize': 24,    # Huge titles
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.edgecolor': '#333333',
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.color': 'black'
    })

    # Layout: 3x3
    fig = plt.figure(figsize=(24, 18), constrained_layout=True)
    # Give the middle column (Metrics) slightly less width, but enough for vertical text
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 0.7, 1])

    fig.suptitle(f"Evaluation Report | {datetime.now().strftime('%Y-%m-%d')}",
                 fontsize=22, y=1.03, fontweight='normal')

    # --- Gradient Line Plot Helper ---
    def plot_gradient_line(ax, x, y, title, ylabel, rising_is_good=True, is_loss=False):
        if is_loss:
            upper = max(1.05, max(y) + 0.1) if len(y) > 0 else 1.0
            ax.set_ylim(0, upper)
        else:
            ax.set_ylim(0, 1.02)

        if len(x) > 0:
            ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Single point case
        if len(x) < 2:
            val = y[0] if len(y) > 0 else 0
            ax.scatter(x, y, color=MAIN_RED, s=150, zorder=5)
            ax.text(x[0], val + (0.02 if not is_loss else -0.05), f"{val:.4f}",
                    color=MAIN_RED, fontweight='bold', ha='center', fontsize=16)
            ax.set_title(title, pad=20)
            ax.set_ylabel(ylabel)
            return

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        diffs = np.diff(y)
        if rising_is_good:
            colors = [MAIN_RED if d >= 0 else 'black' for d in diffs]
        else:
            colors = [MAIN_RED if d <= 0 else 'black' for d in diffs]

        lc = LineCollection(segments, colors=colors, linewidths=4)
        ax.add_collection(lc)

        last_color = colors[-1] if len(colors) > 0 else MAIN_RED
        ax.scatter(x[-1], y[-1], color=last_color, s=150, zorder=5)
        ax.text(x[-1], y[-1] + (0.02 if rising_is_good else -0.05),
                f"{y[-1]:.4f}", color=last_color, fontweight='bold', ha='center', fontsize=16)

        ax.set_title(title, pad=20, fontsize=30, fontweight='bold')
        ax.set_ylabel(ylabel)

    # A. Accuracy (Top Left)
    ax_acc = fig.add_subplot(gs[0, :2])
    epochs = np.arange(1, len(history['val_acc']) + 1)
    plot_gradient_line(ax_acc, epochs, history['val_acc'], 'Accuracy', 'Accuracy', rising_is_good=True)

    # B. ROC Curves (Top Right)
    ax_roc = fig.add_subplot(gs[0, 2])
    y_bin = label_binarize(y_true, classes=range(n_classes))
    for i in range(n_classes):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, lw=3, color=CLS_COLORS[i % 4],
                        label=f'{classes[i]} ({roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.4)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_title('ROC Curves')
    ax_roc.legend(loc='lower right', frameon=False, fontsize=12)

    # C. Confusion Matrix (Mid Left)
    ax_cm = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_true, y_pred)
    cmap_rb = sns.blend_palette(['white', '#dddddd', 'black', MAIN_RED], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_rb, ax=ax_cm, cbar=False,
                xticklabels=classes, yticklabels=classes, square=True,
                annot_kws={"size": 20, "weight": "bold", "color": "white"})

    # Fix annotation colors manually because standard heatmap text color logic might fail with custom cmaps
    for text in ax_cm.texts:
        if int(text.get_text()) > cm.max() / 2: text.set_color('white')
        else: text.set_color('black')

    ax_cm.set_title('Confusion Matrix', fontsize=30, fontweight='bold', pad=20)
    ax_cm.set_ylabel('Actual')
    ax_cm.set_xlabel('Predicted')

    # D. Detailed Metrics Heatmap (Mid Center)
    ax_metrics = fig.add_subplot(gs[1, 1])
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    metrics_data = [[rep[c]['precision'], rep[c]['recall'], rep[c]['f1-score']] for c in classes]
    df_metrics = pd.DataFrame(metrics_data, columns=['Precision', 'Recall', 'F1-score'])

    # Heatmap with dark color scheme (Black background mostly)
    # Using a dark red palette
    cmap_dark = sns.dark_palette(MAIN_RED, as_cmap=True, input='rgb')
    sns.heatmap(df_metrics, annot=True, fmt='.2f', cmap=cmap_dark, ax=ax_metrics, cbar=False,
                annot_kws={"size": 18, "weight": "bold", "color": "white"})

    # Customizing the look to match the photo
    ax_metrics.xaxis.tick_top()
    ax_metrics.set_xticklabels(df_metrics.columns, fontsize=18, fontweight='bold')
    ax_metrics.set_yticks([]) # Hide standard Y ticks

    # Add Giant Vertical Class Labels
    for i, cls in enumerate(classes):
        ax_metrics.text(-0.1, i + 0.5, cls.upper(), fontsize=32, fontweight='bold',
                        rotation=90, va='center', ha='right', transform=ax_metrics.transData)

    # E. Confidence Distribution (Mid Right)
    ax_hist = fig.add_subplot(gs[1, 2])
    max_probs = np.max(y_prob, axis=1)
    correct_mask = (y_pred == y_true)
    bins = np.linspace(0.3, 1, 15) # Start from 0.3 to focus on relevant part

    ax_hist.hist(max_probs[~correct_mask], bins=bins, alpha=0.6, color='black', label='Wrong', edgecolor='white')
    ax_hist.hist(max_probs[correct_mask], bins=bins, alpha=0.6, color=MAIN_RED, label='Correct', edgecolor='white')

    ax_hist.set_title('Confidence Distribution')
    ax_hist.set_xlabel('Confidence Score')
    ax_hist.set_yticks([])
    ax_hist.legend(frameon=False, loc='upper left')
    sns.despine(ax=ax_hist, left=True)

    # F. Loss Graph (Bottom Left)
    ax_loss = fig.add_subplot(gs[2, 0])
    loss_hist = history['train_loss']
    loss_epochs = np.arange(1, len(loss_hist) + 1)
    plot_gradient_line(ax_loss, loss_epochs, loss_hist, 'Train Loss', 'Loss', rising_is_good=False, is_loss=True)

    # G. Precision-Recall Curves (Bottom Right)
    ax_pr = fig.add_subplot(gs[2, 1:])
    for i in range(n_classes):
        if np.sum(y_bin[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            ap = average_precision_score(y_bin[:, i], y_prob[:, i])
            ax_pr.plot(recall, precision, lw=3, color=CLS_COLORS[i % 4],
                       label=f'{classes[i]} (AP={ap:.2f})')
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc='lower left', frameon=False, fontsize=12)

    # Save
    logger.save_figure(fig, "full_report.png")
    plt.close(fig)
    return model, loader, classes, history, 0

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
    'epochs': 1,
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

# %% [CELL] 

run_full_evaluation(model, val_loader, logger, history, class_names, CONFIG)

create_submission(model, class_names, logger, CONFIG)

logger.push_results()