# Exported at 2025-12-16 17:43:21.705109
# Total Cells: 58

# In[1]
import os
import torch
import torch.nn as nn
import pandas as pd
import subprocess
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from torchvision.models import EfficientNet_V2_S_Weights
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import userdata


import kagglehub
os.environ['KAGGLEHUB_CACHE'] = os.path.abspath('/content')
path_filtered = kagglehub.dataset_download("kvdep1/dumplings")

#https://www.kaggle.com/competitions/yummi-classification-fu25/data
path_SUBMISSION_dumpling = kagglehub.competition_download("yummi-classification-fu25")

token = userdata.get('GITHUB_DLTESTWORK_TOKEN') or os.environ.get('GITHUB_DLTESTWORK_TOKEN')

# Конфиг
CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token, # Убедитесь, что переменная установлена
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_SUBMISSION_dumpling}/test', # Путь куда распаковали данные Kaggle
    'batch_size': 32,
    'epochs': 8,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# In[2]
import os
import shutil
import subprocess
from datetime import datetime
from IPython import get_ipython

class GitLogger:
    def __init__(self, repo_url, token, subdir="dumplings"):
        self.repo_name = "DL-TESTWORK"
        self.root_subdir = subdir
        
        # Clean URL and insert token
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        self.auth_url = f"https://{token}@{clean_url}"
        
        # Absolute path to avoid CWD confusion
        self.base_path = os.path.abspath(self.repo_name)
        self.current_exp_folder = None
        
        self._setup_repo()

    def _run(self, cmd, cwd=None, ignore_errors=False):
        if cwd is None:
            cwd = self.base_path
            
        print(f"⚙️ Running: {cmd} | CWD: {cwd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout: print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                print(f"❌ Error executing {cmd}:")
                print(e.stderr)
            else:
                print(f"⚠️ Notice: {cmd} returned non-zero (likely nothing to commit).")

    def _setup_repo(self):
        # 1. Clean existing
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        
        # 2. Clone
        print("📥 Cloning repository...")
        # Clone into the specific folder name
        subprocess.check_call(f"git clone {self.auth_url} {self.repo_name}", shell=True)
        
        # 3. Config
        self._run("git config user.email 'bot@colab.com'")
        self._run("git config user.name 'ColabBot'")

    def start_experiment(self, val_acc):
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = f"exp_acc_{val_acc:.4f}_{date_str}"
        
        # Create full path inside the repo
        self.current_exp_folder = os.path.join(self.base_path, self.root_subdir, folder_name)
        os.makedirs(self.current_exp_folder, exist_ok=True)
        
        print(f"📂 Created experiment folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder:
            print("⚠️ Experiment folder not created.")
            return

        target = os.path.join(self.current_exp_folder, "run_code.py")
        
        # FIX: Access history directly instead of using magic commands with fragile quoting
        ip = get_ipython()
        # Get last 200 inputs. format: [(session, line, source), ...]
        history = ip.history_manager.get_range(limit=200)
        
        try:
            with open(target, 'w', encoding='utf-8') as f:
                f.write(f"# Exported at {datetime.now()}\n\n")
                for _, _, source in history:
                    f.write(source + "\n")
            print(f"💾 Code saved to {target}")
        except Exception as e:
            print(f"❌ Failed to save code: {e}")

    def save_figure(self, fig, filename):
        if not self.current_exp_folder: return
        path = os.path.join(self.current_exp_folder, filename)
        fig.savefig(path, bbox_inches='tight', dpi=100)
        print(f"🖼️ Figure saved: {path}")

    def push_results(self):
        print("🚀 Pushing to GitHub...")
        
        # 1. Pull to sync
        self._run("git pull origin main", ignore_errors=True)
        
        # 2. Add all files
        self._run("git add .")
        
        # 3. Commit (Allow failure if nothing to commit)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._run(f"git commit -m 'Auto-result {timestamp}'", ignore_errors=True)
        
        # 4. Push
        try:
            self._run("git push origin main")
            print("✅ Done! Data is on GitHub.")
        except:
            print("❌ Push failed. Check token permissions or conflicts.")

# In[3]
def get_dataloaders():
    # Аугментация для обучения
    train_tf = transforms.Compose([
        transforms.Resize((384, 384)), # V2-S любит разрешение повыше
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_ds = datasets.ImageFolder(CONFIG['data_dir'], transform=train_tf)

    # Сплит 80/20
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # Важно: валидации нужен свой трансформ без аугментаций
    val_ds.dataset.transform = val_tf

    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    return train_dl, val_dl, full_ds.classes

def build_model(num_classes=4):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    # Заморозка первых слоев (опционально, для ускорения)
    for param in model.features[:4].parameters():
        param.requires_grad = False

    # Замена головы
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(CONFIG['device'])

# In[4]
def train():
    train_loader, val_loader, class_names = get_dataloaders()
    print(f"Classes mapping: {class_names}") # ['gyoza', 'khinkali', 'manti', 'pelmeni'] (alphabetical)

    model = build_model(len(class_names))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['lr'],
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=CONFIG['epochs'])

    best_acc = 0.0
    train_losses=[]
    print("🚀 Training Started...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_losses.append(train_loss/len(train_loader))
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_dumpling_model.pth")

    # Logging to Git
    if CONFIG['token']:
        logger = GitLogger(CONFIG['repo_url'], CONFIG['token'], subdir="dumplings_experiments")
        logger.save_code()

        # 4. Генерируем и сохраняем графики

        # --- График Loss ---
        fig_loss = plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        # plt.plot(val_losses, label='Val Loss') # Если есть
        plt.title(f"Training Loss (Best Acc: {best_acc:.4f})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # Сохраняем через логгер
        logger.save_figure(fig_loss, "loss_plot.png")
        plt.close(fig_loss) # Закрываем, чтобы не висел в памяти

        # --- Confusion Matrix (Пример) ---
        # (Здесь должен быть ваш код построения матрицы, допустим он рисует fig_cm)
        # logger.save_figure(fig_cm, "confusion_matrix.png")

        # 5. Отправляем всё на Github
        logger.push_results()

    return model, class_names

# Запуск
model, classes = train()

# In[5]
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = list(Path(root).glob("*.*")) # Ищет все файлы в папке
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path.name # Возвращаем имя файла (ea2d...jpg)

def create_submission(class_names):
    # Загружаем лучшую модель
    model = build_model(len(class_names))
    model.load_state_dict(torch.load("best_dumpling_model.pth"))
    model.eval()

    test_tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds = TestDataset(CONFIG['test_dir'], test_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    results = []
    print("🔮 Inference...")
    with torch.no_grad():
        for imgs, filenames in tqdm(loader):
            imgs = imgs.to(CONFIG['device'])
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for fname, pred_idx in zip(filenames, preds):
                results.append({
                    'filename': fname,
                    'class': class_names[pred_idx.item()] # Конвертация 0 -> 'gyoza'
                })

    df = pd.DataFrame(results)
    df.to_csv("submission.csv", index=False)
    print("✅ Submission saved: submission.csv")

create_submission(classes)

# In[6]
import os
import shutil
import subprocess
from datetime import datetime

class GitLogger:
    def __init__(self, repo_url, token, subdir="dumplings"):
        self.repo_name = "DL-TESTWORK"
        self.root_subdir = subdir
        # Удаляем "https://" если есть, чтобы корректно вставить токен
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        self.auth_url = f"https://{token}@{clean_url}"

        self.current_exp_folder = None
        self._setup_repo()

    def _run(self, cmd, cwd=None):
        if cwd is None:
            cwd = self.repo_name

        print(f"⚙️ Running: {cmd} in {cwd}")
        try:
            # Используем run для захвата вывода ошибок
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error executing {cmd}:")
            print(e.stderr)
            # Не прерываем выполнение, но сообщаем об ошибке

    def _setup_repo(self):
        # 1. Очистка старой папки
        if os.path.exists(self.repo_name):
            shutil.rmtree(self.repo_name)

        # 2. Клонирование
        print("📥 Cloning repository...")
        self._run(f"git clone {self.auth_url} {self.repo_name}", cwd=".")

        # 3. Настройка user (обязательно для commit)
        self._run("git config user.email 'bot@colab.com'")
        self._run("git config user.name 'ColabBot'")

    def start_experiment(self, val_acc):
        # Формируем имя папки
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = f"exp_acc_{val_acc:.4f}_{date_str}"

        # Полный путь: DL-TESTWORK/dumplings/exp_...
        self.current_exp_folder = os.path.join(self.repo_name, self.root_subdir, folder_name)

        os.makedirs(self.current_exp_folder, exist_ok=True)
        print(f"📂 Created folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder: return
        target = os.path.join(self.current_exp_folder, "run_code.py")
        ipython = get_ipython()
        # Сохраняем последние 200 строк истории
        ipython.run_line_magic('history', f'-l 200 -f "{target}"')
        print(f"💾 Code saved to {target}")

    def save_figure(self, fig, filename):
        if not self.current_exp_folder: return
        path = os.path.join(self.current_exp_folder, filename)
        fig.savefig(path)
        print(f"🖼️ Figure saved: {path}")

    def push_results(self):
        print("🚀 Pushing to GitHub...")
        # 1. Pull перед Push, чтобы избежать конфликтов
        self._run("git pull origin main")

        # 2. Добавляем файлы
        self._run("git add .")

        # 3. Коммит
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._run(f"git commit -m 'Auto-result {timestamp}'")

        # 4. Пуш
        self._run("git push origin main")
        print("✅ Done (check logs above if failed)")

# ==========================================
# ПРОВЕРОЧНЫЙ ЗАПУСК (Вставьте после обучения)
# ==========================================

if CONFIG['token']:
    # Инициализация
    logger = GitLogger(CONFIG['repo_url'], CONFIG['token'], subdir="dumplings")

    # Предположим, что обучение прошло и у нас есть accuracy
    # (Если вы уже обучили модель, используйте реальную переменную best_acc)
    # logger.start_experiment(val_acc=best_acc)

    # Тестовый запуск (если нужно просто проверить работу):
    logger.start_experiment(val_acc=0.9999)

    # 1. Сохраняем код
    logger.save_code()

    # 2. Сохраняем графики (предполагаем, что fig_loss и т.д. у вас уже есть в памяти)
    # Если графиков нет, создадим тестовый:
    try:
        if 'fig_loss' in locals():
            logger.save_figure(fig_loss, "loss_plot.png")
        else:
            # Тестовая картинка
            fig_test = plt.figure()
            plt.plot([0,1], [0,1])
            plt.title("Test Plot")
            logger.save_figure(fig_test, "test_plot.png")
    except Exception as e:
        print(f"Skipping plot save: {e}")

    # 3. Отправка
    logger.push_results()
else:
    print("⚠️ Token is missing in CONFIG")

# In[7]
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, 
                             auc, precision_recall_fscore_support, accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter

def smooth(y, window=5, poly=2):
    if len(y) < window: return y
    return savgol_filter(y, window, poly)

def run_full_evaluation(model, loader, device, logger, history, classes):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    print("Running Inference...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            prob = torch.nn.functional.softmax(out, dim=1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.max(prob, 1)[1].cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    n_classes = len(classes)

    # Setup Style
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("rocket", n_colors=n_classes)
    
    # Master Figure
    fig = plt.figure(figsize=(24, 18), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    # 1. Training Curves (Loss & Accuracy)
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Dual axis for Loss and Acc
    ax1_acc = ax1.twinx()
    
    l1 = ax1.plot(epochs, smooth(history['train_loss']), 'r-', lw=3, label='Train Loss', alpha=0.8)
    l2 = ax1_acc.plot(epochs, history.get('val_acc', []), 'b-', lw=3, label='Val Acc', alpha=0.8)
    
    ax1.set_title("Training Dynamics", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1_acc.set_ylabel("Accuracy")
    
    lines = l1 + l2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='center right')
    ax1.grid(True, alpha=0.3)

    # 2. Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='magma', ax=ax2, 
                xticklabels=classes, yticklabels=classes, cbar=False)
    ax2.set_title("Confusion Matrix (Counts)", fontsize=16, fontweight='bold')
    ax2.set_ylabel('True')
    ax2.set_xlabel('Predicted')

    # 3. ROC Curves (One-vs-Rest)
    ax3 = fig.add_subplot(gs[0, 2])
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color=palette[i], lw=2.5, 
                 label=f'{classes[i]} (AUC={roc_auc:.2f})')
        
    ax3.plot([0, 1], [0, 1], 'k--', lw=1)
    ax3.set_title("ROC Curves", fontsize=16, fontweight='bold')
    ax3.legend(loc="lower right", fontsize=10)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')

    # 4. Per-Class Metrics (Heatmap)
    ax4 = fig.add_subplot(gs[1, :])
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_rep = pd.DataFrame(rep).iloc[:-1, :n_classes].T 
    
    sns.heatmap(df_rep, annot=True, cmap="viridis", fmt=".3f", ax=ax4, linewidths=1)
    ax4.set_title("Class-wise Metrics (Precision, Recall, F1)", fontsize=16, fontweight='bold')

    # 5. Confidence Distribution (Violin Plot)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Gather max probs and correctness
    max_probs = np.max(y_prob, axis=1)
    correct_mask = (y_pred == y_true)
    status = ["Correct" if c else "Wrong" for c in correct_mask]
    
    df_conf = pd.DataFrame({'Confidence': max_probs, 'Status': status, 'Class': [classes[i] for i in y_true]})
    
    sns.violinplot(data=df_conf, x='Class', y='Confidence', hue='Status', 
                   split=True, inner="quart", ax=ax5, palette={"Correct": "g", "Wrong": "r"})
    ax5.set_title("Model Confidence Distribution (Correct vs Wrong)", fontsize=16, fontweight='bold')
    ax5.set_ylim(0, 1.05)

    # Save & Push
    logger.save_figure(fig, "full_evaluation_report.png")
    plt.close(fig)
    
    # CSV Metrics
    df_metrics = pd.DataFrame(rep).transpose()
    df_metrics.to_csv("metrics_report.csv")
    logger._run("git add metrics_report.csv")
    
    logger.push_results()
    print("Visualization generated and pushed.")


history_dict = {'train_loss': train_losses, 'val_acc': val_acc_history}
run_full_evaluation(model, val_loader, CONFIG['device'], logger, history_dict, class_names)

# In[8]
import os
import shutil
import subprocess
from datetime import datetime
from IPython import get_ipython

class GitLogger:
    def __init__(self, repo_url, token, subdir="dumplings"):
        self.repo_name = "DL-TESTWORK"
        self.root_subdir = subdir
        
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        self.auth_url = f"https://{token}@{clean_url}"
        
        self.base_path = os.path.abspath(self.repo_name)
        self.current_exp_folder = None
        
        self._setup_repo()

    def _run(self, cmd, cwd=None, ignore_errors=False):
        if cwd is None:
            cwd = self.base_path
        
        print(f"[RUN] {cmd}")
        try:
            subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                print(f"[ERROR] {cmd} failed:\n{e.stderr}")
                raise e

    def _setup_repo(self):
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        
        print("[INFO] Cloning repository...")
        subprocess.check_call(f"git clone {self.auth_url} {self.repo_name}", shell=True)
        
        self._run("git config user.email 'bot@colab.com'")
        self._run("git config user.name 'ColabBot'")

    def start_experiment(self, val_acc):
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = f"exp_acc_{val_acc:.4f}_{date_str}"
        
        self.current_exp_folder = os.path.join(self.base_path, self.root_subdir, folder_name)
        os.makedirs(self.current_exp_folder, exist_ok=True)
        
        print(f"[INFO] Created experiment folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder:
            raise RuntimeError("Experiment folder not initialized. Call start_experiment() first.")

        target = os.path.join(self.current_exp_folder, "run_code.py")
        ip = get_ipython()
        # Fallback if history is empty
        try:
            history = list(ip.history_manager.get_range(limit=200))
        except:
            history = []
            
        with open(target, 'w', encoding='utf-8') as f:
            f.write(f"# Exported at {datetime.now()}\n\n")
            for _, _, source in history:
                f.write(source + "\n")
        print(f"[INFO] Code saved to {target}")

    def save_figure(self, fig, filename):
        if not self.current_exp_folder:
            raise RuntimeError("Experiment folder not initialized. Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        fig.savefig(path, bbox_inches='tight')
        print(f"[INFO] Figure saved: {path}")

    def push_results(self):
        print("[INFO] Pushing to GitHub...")
        self._run("git pull origin main", ignore_errors=True)
        self._run("git add .")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Ensure commit happens
        try:
            self._run(f"git commit -m 'Auto-result {timestamp}'")
        except subprocess.CalledProcessError:
            print("[WARN] Nothing to commit. Check if files were actually saved.")
            return

        self._run("git push origin main")
        print("[SUCCESS] Data pushed to GitHub.")

# In[9]
# Execution Sequence
# 1. Init
logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

# 2. Start Experiment (Mandatory)
# Use actual variable or dummy value
current_acc = best_acc if 'best_acc' in locals() else 0.0 
logger.start_experiment(val_acc=current_acc)

# 3. Save artifacts
logger.save_code()
# call your visualization function here using 'logger'
# run_full_evaluation(..., logger=logger, ...)

# 4. Push
logger.push_results()

# In[10]
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
                             auc, accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter
import kagglehub
from google.colab import userdata

# In[11]
token = userdata.get('GITHUB_DLTESTWORK_TOKEN') or os.environ.get('GITHUB_DLTESTWORK_TOKEN')
path_filtered = kagglehub.dataset_download("kvdep1/dumplings")
path_sub = kagglehub.competition_download("yummi-classification-fu25")

CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test',
    'batch_size': 32,
    'epochs': 8,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# In[12]
class GitLogger:
    def __init__(self, repo_url, token, subdir="dumplings"):
        self.repo_name = "DL-TESTWORK"
        self.root_subdir = subdir
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        self.auth_url = f"https://{token}@{clean_url}"
        self.base_path = os.path.abspath(self.repo_name)
        self.current_exp_folder = None
        self._setup_repo()

    def _run(self, cmd, cwd=None, ignore_errors=False):
        if cwd is None: cwd = self.base_path
        print(f"[RUN] {cmd}")
        try:
            subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                print(f"[ERROR] {cmd} failed:\n{e.stderr}")
                raise e

    def _setup_repo(self):
        if os.path.exists(self.base_path): shutil.rmtree(self.base_path)
        print("[INFO] Cloning repository...")
        subprocess.check_call(f"git clone {self.auth_url} {self.repo_name}", shell=True)
        self._run("git config user.email 'bot@colab.com'")
        self._run("git config user.name 'ColabBot'")

    def start_experiment(self, val_acc):
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = f"exp_acc_{val_acc:.4f}_{date_str}"
        self.current_exp_folder = os.path.join(self.base_path, self.root_subdir, folder_name)
        os.makedirs(self.current_exp_folder, exist_ok=True)
        print(f"[INFO] Created experiment folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        target = os.path.join(self.current_exp_folder, "run_code.py")
        ip = get_ipython()
        try: history = list(ip.history_manager.get_range(limit=200))
        except: history = []
        with open(target, 'w', encoding='utf-8') as f:
            f.write(f"# Exported at {datetime.now()}\n\n")
            for _, _, source in history: f.write(source + "\n")
        print(f"[INFO] Code saved to {target}")

    def save_figure(self, fig, filename):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        fig.savefig(path, bbox_inches='tight')
        print(f"[INFO] Figure saved: {path}")
    
    def save_csv(self, df, filename):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        df.to_csv(path, index=False)
        print(f"[INFO] CSV saved: {path}")

    def push_results(self):
        print("[INFO] Pushing to GitHub...")
        self._run("git pull origin main", ignore_errors=True)
        self._run("git add .")
        try: self._run(f"git commit -m 'Auto-result {datetime.now().strftime('%H:%M:%S')}'")
        except: print("[WARN] Nothing to commit.")
        self._run("git push origin main")
        print("[SUCCESS] Data pushed.")

# In[13]
def get_dataloaders():
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_ds = datasets.ImageFolder(CONFIG['data_dir'], transform=tf)
    train_sz = int(0.8 * len(full_ds))
    val_sz = len(full_ds) - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz])
    
    # Simple loader setup
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    return train_dl, val_dl, full_ds.classes

def build_model(num_classes):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(CONFIG['device'])

# In[14]
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

# In[15]
def run_full_evaluation(model, loader, logger, history, classes):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    print("Running Evaluation...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(CONFIG['device'])
            out = model(x)
            prob = torch.nn.functional.softmax(out, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.max(prob, 1)[1].cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
            
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # A. Training Curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = ax1.twinx()
    ax1.plot(history['train_loss'], 'r-', label='Loss')
    ax2.plot(history['val_acc'], 'b-', label='Acc')
    ax1.set_title("Training Dynamics")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    
    # B. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, xticklabels=classes, yticklabels=classes)
    ax3.set_title("Confusion Matrix")
    
    # C. ROC
    ax4 = fig.add_subplot(gs[0, 2])
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax4.plot(fpr, tpr, label=f'{classes[i]}')
    ax4.plot([0,1],[0,1],'k--')
    ax4.legend()
    ax4.set_title("ROC Curves")
    
    # D. Metrics Table
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis('off')
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_rep = pd.DataFrame(rep).transpose().round(3)
    table = ax5.table(cellText=df_rep.values, colLabels=df_rep.columns, rowLabels=df_rep.index, loc='center', cellLoc='center')
    table.scale(1, 2)
    ax5.set_title("Classification Report")

    # Save
    logger.save_figure(fig, "evaluation_report.png")
    plt.close(fig)
    logger.save_csv(df_rep, "metrics.csv")

# In[16]
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = list(Path(root).glob("*.*"))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path.name

def create_submission(model, classes, logger):
    test_tf = transforms.Compose([
        transforms.Resize((384, 384)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = TestDataset(CONFIG['test_dir'], test_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    results = []
    print("Generating Submission...")
    model.eval()
    with torch.no_grad():
        for x, fnames in loader:
            x = x.to(CONFIG['device'])
            _, preds = torch.max(model(x), 1)
            for f, p in zip(fnames, preds):
                results.append({'filename': f, 'class': classes[p.item()]})
    
    df = pd.DataFrame(results)
    logger.save_csv(df, "submission.csv")

# In[17]
logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

# 2. Train Model (Returns history and vars needed for next steps)
model, val_loader, class_names, history, best_acc = train()

# 3. Create Experiment Folder (After we have best_acc)
logger.start_experiment(best_acc)

# 4. Save Code
logger.save_code()

# 5. Run Vis & Save Plots
run_full_evaluation(model, val_loader, logger, history, class_names)

# 6. Create & Save Submission
create_submission(model, class_names, logger)

# 7. Push Everything
logger.push_results()

# In[18]
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
                             auc, accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter
import kagglehub
from google.colab import userdata

# In[19]
token = userdata.get('GITHUB_DLTESTWORK_TOKEN') or os.environ.get('GITHUB_DLTESTWORK_TOKEN')
path_filtered = kagglehub.dataset_download("kvdep1/dumplings")
path_sub = kagglehub.competition_download("yummi-classification-fu25")

# In[20]
class GitLogger:
    def __init__(self, repo_url, token, subdir="dumplings"):
        self.repo_name = "DL-TESTWORK"
        self.root_subdir = subdir
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        self.auth_url = f"https://{token}@{clean_url}"
        self.base_path = os.path.abspath(self.repo_name)
        self.current_exp_folder = None
        self._setup_repo()

    def _run(self, cmd, cwd=None, ignore_errors=False):
        if cwd is None: cwd = self.base_path
        print(f"[RUN] {cmd}")
        try:
            subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                print(f"[ERROR] {cmd} failed:\n{e.stderr}")
                raise e

    def _setup_repo(self):
        if os.path.exists(self.base_path): shutil.rmtree(self.base_path)
        print("[INFO] Cloning repository...")
        subprocess.check_call(f"git clone {self.auth_url} {self.repo_name}", shell=True)
        self._run("git config user.email 'bot@colab.com'")
        self._run("git config user.name 'ColabBot'")

    def start_experiment(self, val_acc):
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = f"exp_acc_{val_acc:.4f}_{date_str}"
        self.current_exp_folder = os.path.join(self.base_path, self.root_subdir, folder_name)
        os.makedirs(self.current_exp_folder, exist_ok=True)
        print(f"[INFO] Created experiment folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        target = os.path.join(self.current_exp_folder, "run_code.py")
        ip = get_ipython()
        try: history = list(ip.history_manager.get_range(limit=200))
        except: history = []
        with open(target, 'w', encoding='utf-8') as f:
            f.write(f"# Exported at {datetime.now()}\n\n")
            for _, _, source in history: f.write(source + "\n")
        print(f"[INFO] Code saved to {target}")

    def save_figure(self, fig, filename):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        fig.savefig(path, bbox_inches='tight')
        print(f"[INFO] Figure saved: {path}")
    
    def save_csv(self, df, filename):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        df.to_csv(path, index=False)
        print(f"[INFO] CSV saved: {path}")

    def push_results(self):
        print("[INFO] Pushing to GitHub...")
        self._run("git pull origin main", ignore_errors=True)
        self._run("git add .")
        try: self._run(f"git commit -m 'Auto-result {datetime.now().strftime('%H:%M:%S')}'")
        except: print("[WARN] Nothing to commit.")
        self._run("git push origin main")
        print("[SUCCESS] Data pushed.")

# In[21]
def get_dataloaders():
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_ds = datasets.ImageFolder(CONFIG['data_dir'], transform=tf)
    train_sz = int(0.8 * len(full_ds))
    val_sz = len(full_ds) - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz])
    
    # Simple loader setup
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    return train_dl, val_dl, full_ds.classes

def build_model(num_classes):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(CONFIG['device'])

# In[22]
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

# In[23]
def run_full_evaluation(model, loader, logger, history, classes):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    print("Running Evaluation...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(CONFIG['device'])
            out = model(x)
            prob = torch.nn.functional.softmax(out, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.max(prob, 1)[1].cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
            
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # A. Training Curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = ax1.twinx()
    ax1.plot(history['train_loss'], 'r-', label='Loss')
    ax2.plot(history['val_acc'], 'b-', label='Acc')
    ax1.set_title("Training Dynamics")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    
    # B. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, xticklabels=classes, yticklabels=classes)
    ax3.set_title("Confusion Matrix")
    
    # C. ROC
    ax4 = fig.add_subplot(gs[0, 2])
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax4.plot(fpr, tpr, label=f'{classes[i]}')
    ax4.plot([0,1],[0,1],'k--')
    ax4.legend()
    ax4.set_title("ROC Curves")
    
    # D. Metrics Table
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis('off')
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_rep = pd.DataFrame(rep).transpose().round(3)
    table = ax5.table(cellText=df_rep.values, colLabels=df_rep.columns, rowLabels=df_rep.index, loc='center', cellLoc='center')
    table.scale(1, 2)
    ax5.set_title("Classification Report")

    # Save
    logger.save_figure(fig, "evaluation_report.png")
    plt.close(fig)
    logger.save_csv(df_rep, "metrics.csv")

# In[24]
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = list(Path(root).glob("*.*"))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path.name

def create_submission(model, classes, logger):
    test_tf = transforms.Compose([
        transforms.Resize((384, 384)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = TestDataset(CONFIG['test_dir'], test_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    results = []
    print("Generating Submission...")
    model.eval()
    with torch.no_grad():
        for x, fnames in loader:
            x = x.to(CONFIG['device'])
            _, preds = torch.max(model(x), 1)
            for f, p in zip(fnames, preds):
                results.append({'filename': f, 'class': classes[p.item()]})
    
    df = pd.DataFrame(results)
    logger.save_csv(df, "submission.csv")

# In[25]
CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test',
    'batch_size': 32,
    'epochs': 1,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

# 2. Train Model (Returns history and vars needed for next steps)
model, val_loader, class_names, history, best_acc = train()

# 3. Create Experiment Folder (After we have best_acc)
logger.start_experiment(best_acc)

# 4. Save Code
logger.save_code()

# 5. Run Vis & Save Plots
run_full_evaluation(model, val_loader, logger, history, class_names)

# 6. Create & Save Submission
create_submission(model, class_names, logger)

# 7. Push Everything
logger.push_results()

# In[26]
CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test',
    'batch_size': 16,
    'epochs': 1,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

model, val_loader, class_names, history, best_acc = train()

# 3. Create Experiment Folder (After we have best_acc)
logger.start_experiment(best_acc)

# 4. Save Code
logger.save_code()

# 5. Run Vis & Save Plots
run_full_evaluation(model, val_loader, logger, history, class_names)

# 6. Create & Save Submission
create_submission(model, class_names, logger)

# 7. Push Everything
logger.push_results()

# In[27]
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
                             auc, accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter
import kagglehub
from google.colab import userdata
import gc

# In[28]
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
                             auc, accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter
import kagglehub
from google.colab import userdata
import gc

# In[29]
token = userdata.get('GITHUB_DLTESTWORK_TOKEN') or os.environ.get('GITHUB_DLTESTWORK_TOKEN')
path_filtered = kagglehub.dataset_download("kvdep1/dumplings")
path_sub = kagglehub.competition_download("yummi-classification-fu25")

# In[30]
class GitLogger:
    def __init__(self, repo_url, token, subdir="dumplings"):
        self.repo_name = "DL-TESTWORK"
        self.root_subdir = subdir
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        self.auth_url = f"https://{token}@{clean_url}"
        self.base_path = os.path.abspath(self.repo_name)
        self.current_exp_folder = None
        self._setup_repo()

    def _run(self, cmd, cwd=None, ignore_errors=False):
        if cwd is None: cwd = self.base_path
        print(f"[RUN] {cmd}")
        try:
            subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                print(f"[ERROR] {cmd} failed:\n{e.stderr}")
                raise e

    def _setup_repo(self):
        if os.path.exists(self.base_path): shutil.rmtree(self.base_path)
        print("[INFO] Cloning repository...")
        subprocess.check_call(f"git clone {self.auth_url} {self.repo_name}", shell=True)
        self._run("git config user.email 'bot@colab.com'")
        self._run("git config user.name 'ColabBot'")

    def start_experiment(self, val_acc):
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = f"exp_acc_{val_acc:.4f}_{date_str}"
        self.current_exp_folder = os.path.join(self.base_path, self.root_subdir, folder_name)
        os.makedirs(self.current_exp_folder, exist_ok=True)
        print(f"[INFO] Created experiment folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        target = os.path.join(self.current_exp_folder, "run_code.py")
        ip = get_ipython()
        try: history = list(ip.history_manager.get_range(limit=200))
        except: history = []
        with open(target, 'w', encoding='utf-8') as f:
            f.write(f"# Exported at {datetime.now()}\n\n")
            for _, _, source in history: f.write(source + "\n")
        print(f"[INFO] Code saved to {target}")

    def save_figure(self, fig, filename):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        fig.savefig(path, bbox_inches='tight')
        print(f"[INFO] Figure saved: {path}")
    
    def save_csv(self, df, filename):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        df.to_csv(path, index=False)
        print(f"[INFO] CSV saved: {path}")

    def push_results(self):
        print("[INFO] Pushing to GitHub...")
        self._run("git pull origin main", ignore_errors=True)
        self._run("git add .")
        try: self._run(f"git commit -m 'Auto-result {datetime.now().strftime('%H:%M:%S')}'")
        except: print("[WARN] Nothing to commit.")
        self._run("git push origin main")
        print("[SUCCESS] Data pushed.")

# In[31]
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

# In[32]
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

# In[33]
def run_full_evaluation(model, loader, logger, history, classes):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    print("Running Evaluation...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(CONFIG['device'])
            out = model(x)
            prob = torch.nn.functional.softmax(out, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.max(prob, 1)[1].cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
            
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # A. Training Curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = ax1.twinx()
    ax1.plot(history['train_loss'], 'r-', label='Loss')
    ax2.plot(history['val_acc'], 'b-', label='Acc')
    ax1.set_title("Training Dynamics")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    
    # B. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, xticklabels=classes, yticklabels=classes)
    ax3.set_title("Confusion Matrix")
    
    # C. ROC
    ax4 = fig.add_subplot(gs[0, 2])
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax4.plot(fpr, tpr, label=f'{classes[i]}')
    ax4.plot([0,1],[0,1],'k--')
    ax4.legend()
    ax4.set_title("ROC Curves")
    
    # D. Metrics Table
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis('off')
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_rep = pd.DataFrame(rep).transpose().round(3)
    table = ax5.table(cellText=df_rep.values, colLabels=df_rep.columns, rowLabels=df_rep.index, loc='center', cellLoc='center')
    table.scale(1, 2)
    ax5.set_title("Classification Report")

    # Save
    logger.save_figure(fig, "evaluation_report.png")
    plt.close(fig)
    logger.save_csv(df_rep, "metrics.csv")

# In[34]
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = list(Path(root).glob("*.*"))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path.name

def create_submission(model, classes, logger):
    test_tf = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = TestDataset(CONFIG['test_dir'], test_tf)
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    results = []
    print("Generating Submission...")
    model.eval()
    with torch.no_grad():
        for x, fnames in loader:
            x = x.to(CONFIG['device'])
            _, preds = torch.max(model(x), 1)
            for f, p in zip(fnames, preds):
                results.append({'filename': f, 'class': classes[p.item()]})
    
    df = pd.DataFrame(results)
    logger.save_csv(df, "submission.csv")

# In[35]
def free_gpu():
    if 'model' in globals(): del globals()['model']
    if 'optimizer' in globals(): del globals()['optimizer']
    if 'scheduler' in globals(): del globals()['scheduler']
    gc.collect()
    torch.cuda.empty_cache()
free_gpu()

# In[36]
CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test',
    'batch_size': 16,
    'img_size' : 224
    'epochs': 1,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

model, val_loader, class_names, history, best_acc = train()

logger.start_experiment(best_acc)

logger.save_code()

run_full_evaluation(model, val_loader, logger, history, class_names)

create_submission(model, class_names, logger)

logger.push_results()

# In[37]
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
                             auc, accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter
import kagglehub
from google.colab import userdata
import gc

# In[38]
token = userdata.get('GITHUB_DLTESTWORK_TOKEN') or os.environ.get('GITHUB_DLTESTWORK_TOKEN')
path_filtered = kagglehub.dataset_download("kvdep1/dumplings")
path_sub = kagglehub.competition_download("yummi-classification-fu25")

# In[39]
class GitLogger:
    def __init__(self, repo_url, token, subdir="dumplings"):
        self.repo_name = "DL-TESTWORK"
        self.root_subdir = subdir
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        self.auth_url = f"https://{token}@{clean_url}"
        self.base_path = os.path.abspath(self.repo_name)
        self.current_exp_folder = None
        self._setup_repo()

    def _run(self, cmd, cwd=None, ignore_errors=False):
        if cwd is None: cwd = self.base_path
        print(f"[RUN] {cmd}")
        try:
            subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                print(f"[ERROR] {cmd} failed:\n{e.stderr}")
                raise e

    def _setup_repo(self):
        if os.path.exists(self.base_path): shutil.rmtree(self.base_path)
        print("[INFO] Cloning repository...")
        subprocess.check_call(f"git clone {self.auth_url} {self.repo_name}", shell=True)
        self._run("git config user.email 'bot@colab.com'")
        self._run("git config user.name 'ColabBot'")

    def start_experiment(self, val_acc):
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = f"exp_acc_{val_acc:.4f}_{date_str}"
        self.current_exp_folder = os.path.join(self.base_path, self.root_subdir, folder_name)
        os.makedirs(self.current_exp_folder, exist_ok=True)
        print(f"[INFO] Created experiment folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        target = os.path.join(self.current_exp_folder, "run_code.py")
        ip = get_ipython()
        try: history = list(ip.history_manager.get_range(limit=200))
        except: history = []
        with open(target, 'w', encoding='utf-8') as f:
            f.write(f"# Exported at {datetime.now()}\n\n")
            for _, _, source in history: f.write(source + "\n")
        print(f"[INFO] Code saved to {target}")

    def save_figure(self, fig, filename):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        fig.savefig(path, bbox_inches='tight')
        print(f"[INFO] Figure saved: {path}")
    
    def save_csv(self, df, filename):
        if not self.current_exp_folder: raise RuntimeError("Call start_experiment() first.")
        path = os.path.join(self.current_exp_folder, filename)
        df.to_csv(path, index=False)
        print(f"[INFO] CSV saved: {path}")

    def push_results(self):
        print("[INFO] Pushing to GitHub...")
        self._run("git pull origin main", ignore_errors=True)
        self._run("git add .")
        try: self._run(f"git commit -m 'Auto-result {datetime.now().strftime('%H:%M:%S')}'")
        except: print("[WARN] Nothing to commit.")
        self._run("git push origin main")
        print("[SUCCESS] Data pushed.")

# In[40]
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

# In[41]
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

# In[42]
def run_full_evaluation(model, loader, logger, history, classes):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    print("Running Evaluation...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(CONFIG['device'])
            out = model(x)
            prob = torch.nn.functional.softmax(out, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.max(prob, 1)[1].cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
            
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # A. Training Curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = ax1.twinx()
    ax1.plot(history['train_loss'], 'r-', label='Loss')
    ax2.plot(history['val_acc'], 'b-', label='Acc')
    ax1.set_title("Training Dynamics")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    
    # B. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, xticklabels=classes, yticklabels=classes)
    ax3.set_title("Confusion Matrix")
    
    # C. ROC
    ax4 = fig.add_subplot(gs[0, 2])
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax4.plot(fpr, tpr, label=f'{classes[i]}')
    ax4.plot([0,1],[0,1],'k--')
    ax4.legend()
    ax4.set_title("ROC Curves")
    
    # D. Metrics Table
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis('off')
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_rep = pd.DataFrame(rep).transpose().round(3)
    table = ax5.table(cellText=df_rep.values, colLabels=df_rep.columns, rowLabels=df_rep.index, loc='center', cellLoc='center')
    table.scale(1, 2)
    ax5.set_title("Classification Report")

    # Save
    logger.save_figure(fig, "evaluation_report.png")
    plt.close(fig)
    logger.save_csv(df_rep, "metrics.csv")

# In[43]
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = list(Path(root).glob("*.*"))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path.name

def create_submission(model, classes, logger):
    test_tf = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = TestDataset(CONFIG['test_dir'], test_tf)
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    results = []
    print("Generating Submission...")
    model.eval()
    with torch.no_grad():
        for x, fnames in loader:
            x = x.to(CONFIG['device'])
            _, preds = torch.max(model(x), 1)
            for f, p in zip(fnames, preds):
                results.append({'filename': f, 'class': classes[p.item()]})
    
    df = pd.DataFrame(results)
    logger.save_csv(df, "submission.csv")

# In[44]
def free_gpu():
    if 'model' in globals(): del globals()['model']
    if 'optimizer' in globals(): del globals()['optimizer']
    if 'scheduler' in globals(): del globals()['scheduler']
    gc.collect()
    torch.cuda.empty_cache()
free_gpu()

# In[45]
CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test',
    'batch_size': 16,
    'img_size' : 224,
    'epochs': 1,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

model, val_loader, class_names, history, best_acc = train()

logger.start_experiment(best_acc)

logger.save_code()

run_full_evaluation(model, val_loader, logger, history, class_names)

create_submission(model, class_names, logger)

logger.push_results()

# In[46]
path_sub

# In[47]
CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test/test/images',
    'batch_size': 16,
    'img_size' : 224,
    'epochs': 1,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

# 2. Experiment (Ensure best_acc is defined from training)
# If training cell isn't run in this session, uncomment below:
# best_acc = 0.8559 
logger.start_experiment(best_acc)

# 3. Save Code (Robust method)
logger.save_code()

# 4. Generate Elegant Report
# We need y_true and y_pred from validation to do this.
# Re-running inference on val_loader if variables are lost:
if 'val_loader' in locals() and 'model' in locals():
    print("Collecting validation metrics for report...")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(CONFIG['device'])
            out = model(x)
            _, p = torch.max(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(p.cpu().numpy())
            
    visualize_elegant(y_true, y_pred, class_names, logger)
else:
    print("⚠️ Model/Loader not in memory. Run training cell first to generate report.")

# 5. Submission
# Ensure CONFIG['test_dir'] is set to the folder containing images
# Example: '/root/.cache/kagglehub/.../test/test/images'
create_submission(model, class_names, logger, CONFIG)

# 6. Push
logger.push_results()

# In[48]
CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test/test/images',
    'batch_size': 16,
    'img_size' : 224,
    'epochs': 1,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

model, val_loader, class_names, history, best_acc = train()

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

# 2. Experiment (Ensure best_acc is defined from training)
# If training cell isn't run in this session, uncomment below:
# best_acc = 0.8559 
logger.start_experiment(best_acc)

# 3. Save Code (Robust method)
logger.save_code()

# 4. Generate Elegant Report
# We need y_true and y_pred from validation to do this.
# Re-running inference on val_loader if variables are lost:
if 'val_loader' in locals() and 'model' in locals():
    print("Collecting validation metrics for report...")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(CONFIG['device'])
            out = model(x)
            _, p = torch.max(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(p.cpu().numpy())
            
    visualize_elegant(y_true, y_pred, class_names, logger)
else:
    print("⚠️ Model/Loader not in memory. Run training cell first to generate report.")

# 5. Submission
# Ensure CONFIG['test_dir'] is set to the folder containing images
# Example: '/root/.cache/kagglehub/.../test/test/images'
create_submission(model, class_names, logger, CONFIG)

# 6. Push
logger.push_results()

# In[49]
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
                             auc, accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.signal import savgol_filter
import kagglehub
from google.colab import userdata
import gc

# In[50]
token = userdata.get('GITHUB_DLTESTWORK_TOKEN') or os.environ.get('GITHUB_DLTESTWORK_TOKEN')
path_filtered = kagglehub.dataset_download("kvdep1/dumplings")
path_sub = kagglehub.competition_download("yummi-classification-fu25")

# In[51]
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
        # Quote paths to handle special chars like | or :
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
        # Format: 0.8559|20:32_16.12.2025
        # Note: Colon (:) is fine in Linux/Colab filenames but invalid in Windows.
        now = datetime.now()
        date_str = now.strftime("%H:%M_%d.%m.%Y")
        
        # We store val_acc for the commit message later
        self.last_val_acc = val_acc 
        
        folder_name = f"{val_acc:.4f}|{date_str}"
        self.current_exp_folder = os.path.join(self.base_path, self.root_subdir, folder_name)
        
        # Create dir (Linux handles pipes and colons fine)
        os.makedirs(self.current_exp_folder, exist_ok=True)
        print(f"[INFO] Created folder: {self.current_exp_folder}")
        return self.current_exp_folder

    def save_code(self):
        if not self.current_exp_folder: return
        target = os.path.join(self.current_exp_folder, "run_code.py")
        
        # Access global input history list directly
        # _ih is a built-in list in IPython containing all input strings
        raw_history = globals().get('_ih', [])
        
        with open(target, 'w', encoding='utf-8') as f:
            f.write(f"# Exported at {datetime.now()}\n")
            f.write(f"# Total Cells: {len(raw_history)}\n\n")
            for i, code in enumerate(raw_history):
                if code.strip(): # Skip empty cells
                    f.write(f"# In[{i}]\n{code}\n\n")
        print(f"[INFO] Code saved ({len(raw_history)} cells).")

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
        
        # Commit message is ONLY the val accuracy
        msg = f"{self.last_val_acc:.4f}"
        self._run(f"git commit -m '{msg}'")
        self._run("git push origin main")
        print("[SUCCESS] Pushed.")

# In[52]
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

# In[53]
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

# In[54]
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

# In[55]
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = list(Path(root).glob("*.*"))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path.name

def create_submission(model, classes, logger):
    test_tf = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = TestDataset(CONFIG['test_dir'], test_tf)
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    results = []
    print("Generating Submission...")
    model.eval()
    with torch.no_grad():
        for x, fnames in loader:
            x = x.to(CONFIG['device'])
            _, preds = torch.max(model(x), 1)
            for f, p in zip(fnames, preds):
                results.append({'filename': f, 'class': classes[p.item()]})
    
    df = pd.DataFrame(results)
    logger.save_csv(df, "submission.csv")

# In[56]
def free_gpu():
    if 'model' in globals(): del globals()['model']
    if 'optimizer' in globals(): del globals()['optimizer']
    if 'scheduler' in globals(): del globals()['scheduler']
    gc.collect()
    torch.cuda.empty_cache()
free_gpu()

# In[57]
CONFIG = {
    'repo_url': 'github.com/kvdep/DL-TESTWORK.git',
    'token': token,
    'data_dir': f'{path_filtered}/dataset_filtered',
    'test_dir': f'{path_sub}/test/test/images',
    'batch_size': 16,
    'img_size' : 224,
    'epochs': 1,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])

model, val_loader, class_names, history, best_acc = train()

logger = GitLogger(CONFIG['repo_url'], CONFIG['token'])


logger.start_experiment(best_acc)


logger.save_code()

if 'val_loader' in locals() and 'model' in locals():
    print("Collecting validation metrics for report...")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(CONFIG['device'])
            out = model(x)
            _, p = torch.max(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(p.cpu().numpy())
            
    visualize_elegant(y_true, y_pred, class_names, logger)
else:
    print("⚠️ Model/Loader not in memory. Run training cell first to generate report.")


create_submission(model, class_names, logger, CONFIG)

# 6. Push
logger.push_results()

