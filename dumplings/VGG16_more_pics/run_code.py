
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Image as dispImage
from IPython.display import clear_output
import os
import pytorch_lightning as pl
import zipfile
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint,EarlyStopping 
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from functools import reduce
import numpy as np
import random
import pandas as pd
import torchvision.models as models
import time
import subprocess
import shutil
import tempfile
import os
import dotenv
import json
import sys
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             auc, precision_recall_curve, average_precision_score, accuracy_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns

# ==================================================

DATA_FOLDER = Path('data') /'yummi-classification-fu25'
TEST_PATH = DATA_FOLDER / 'test'
TRAIN_PATH = DATA_FOLDER / 'train'

# ==================================================

# for i in TRAIN_PATH.rglob('*.jpg'):
#     display(dispImage(i), str(i))
#     if input('DELETE image? Y/N :').lower() == 'y':
#         os.remove(str(i))
#     clear_output(True)

# ==================================================

class DataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, zip_path: str, batch_size: int = 64, num_workers: int = 7, random_state = None):
        super().__init__()
        self.data_path = data_path
        self.zip_path = zip_path
        self.batch_size = batch_size
        self.transform = None
        self.train_dataset = None
        self.test_dataset = None
        self.class_names = None
        self.num_classes = 0
        self.num_workers = num_workers
        self.random_state = random_state

    def prepare_data(self):
        if not os.path.exists(self.data_path):
            print(f"Распаковка архива {self.zip_path}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            print("Архив успешно распакован.")
        else:
            print(f"Директория {self.data_path} уже существует.")

    def setup(self, stage: str = None):
        # 1. Предварительная трансформация для расчета статистики
        pre_transform = T.Compose([
            T.Resize((300, 300)),
            T.ToImage(),  
            T.ToDtype(torch.float32, scale=True) # Теперь работает с тензором
        ])
        
        full_dataset_for_stats = ImageFolder(root=self.data_path, transform=pre_transform)
        
        # # 2. Разбиение на train/test
        # train_size = int(0.8 * len(full_dataset_for_stats))
        # test_size = len(full_dataset_for_stats) - train_size
        # generator = torch.Generator().manual_seed(42)
        # train_subset_for_stats, _ = random_split(full_dataset_for_stats, [train_size, test_size], generator=generator)

        # 3. Расчет статистики по обучающей выборке
        print("\nРасчет статистики для нормализации...")
        loader = DataLoader(full_dataset_for_stats, batch_size=self.batch_size,)
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        nb_samples = 0
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            nb_samples += batch_samples
        
        mean /= nb_samples
        std /= nb_samples
        print(f"Рассчитанное среднее (mean): {mean}")
        print(f"Рассчитанное стандартное отклонение (std): {std}")

        #4. Финальная трансформация с нормализацией
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std)
        ])
            # Трансформации для ОБУЧАЮЩЕГО набора данных (с аугментацией)
        

            
        self.train_transform1 = T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(p=0.5), # Случайное горизонтальное отражение
            T.RandomRotation(degrees=15), # Случайный поворот на 15 градусов
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Случайное изменение яркости/контраста
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std)
        ])
        
        self.train_transform2 = T.Compose([
            # RandomResizedCrop заставляет модель смотреть на детали, а не только на общую форму
            T.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)), 
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std)
        ])

        # Трансформации для ТЕСТОВОГО набора (без аугментации!)
        self.test_transform = T.Compose([
            T.Resize((128, 128)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std)
        ])
        

        #4. Финальная трансформация с нормализацией
        self.transform = T.Compose([
            T.Resize((300, 300)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std)
        ])
            # Трансформации для ОБУЧАЮЩЕГО набора данных (с аугментацией)
        

            
        self.train_transform1 = T.Compose([
            T.Resize((300, 300)),
            T.RandomHorizontalFlip(p=0.5), # Случайное горизонтальное отражение
            T.RandomRotation(degrees=15), # Случайный поворот на 15 градусов
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Случайное изменение яркости/контраста
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std)
        ])
        
        self.train_transform2 = T.Compose([
            # RandomResizedCrop заставляет модель смотреть на детали, а не только на общую форму
            T.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0)), 
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std)
        ])

        # Трансформации для ТЕСТОВОГО набора (без аугментации!)
        self.test_transform = T.Compose([
            T.Resize((300, 300)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std)
        ])
        
        
        self.std = std
        self.mean = mean
        # 5. Создание финальных датасетов с разными трансформациями
        dataset_train1 = ImageFolder(root=self.data_path, transform=self.train_transform1)
        dataset_train2 = ImageFolder(root=self.data_path, transform=self.train_transform2)
        dataset_train3 = ImageFolder(root=self.data_path, transform=self.test_transform)
        dataset_test = ImageFolder(root=self.data_path, transform=self.test_transform)
        
        self.class_names = dataset_train1.classes
        self.num_classes = len(self.class_names)
        
        
        
        # Важно! Мы должны разделить один и тот же "исходный" датасет
        # random_split не может работать с двумя разными объектами датасетов
        full_dataset_final = ImageFolder(root=self.data_path) # Без трансформаций
        train_size = int(0.8 * len(full_dataset_final))
        test_size = len(full_dataset_final) - train_size
        self.dataset = full_dataset_final
        
        generator = torch.Generator().manual_seed(42)
        # train_indices, test_indices = random_split(range(len(full_dataset_final)), [train_size, test_size], generator=generator)
        if self.random_state==None:
            rs = np.random.randint(0,1000)
        else:
            rs = self.random_state
        train_indices, test_indices = train_test_split(range(len(full_dataset_final)), stratify=full_dataset_final.targets, test_size=0.2, random_state=rs)
        print(rs)

        
        # Теперь применяем нужные трансформации к нужным подвыборкам
        # self.train_dataset = torch.utils.data.Subset(dataset_train, train_indices.indices)
        # self.test_dataset = torch.utils.data.Subset(dataset_test, test_indices.indices)
        subset1 = Subset(dataset_train1, train_indices)
        subset2= Subset(dataset_train2, train_indices)
        subset3 = Subset(dataset_train3, train_indices)
        
        self.train_dataset =  ConcatDataset([subset1, subset2, subset3])

        self.test_dataset = Subset(dataset_test, test_indices)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return self.test_dataloader()

# ==================================================

zip_path = r"C:\Projects\FU\Course_3\DL\data\yummi-classification-fu25\dataset_filtered v2.zip"
dm = DataModule(data_path=TRAIN_PATH/'train', zip_path=zip_path, batch_size=64)

dm.prepare_data()
dm.setup()


print(f"Размер обучающей выборки: {len(dm.train_dataset)}")
print(f"Размер тестовой выборки: {len(dm.test_dataset)}")

image_sample, _ = dm.train_dataset[0]


print(f"\nОбщее количество картинок в датасете: {len(dm.train_dataset) + len(dm.test_dataset)}")
print(f"Размер одной картинки (тензора): {image_sample.shape}")
print(f"Количество уникальных классов: {dm.num_classes}")
print("Названия классов:", dm.class_names)

# ==================================================

class PartialFineTuneVGG(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4): # Используем меньший learning rate
        super().__init__()
        self.save_hyperparameters()

        weights = models.VGG16_Weights.DEFAULT
        self.model = models.vgg16(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False
            
        for param in self.model.features[28].parameters():
             param.requires_grad = True
                
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_loss':[]}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)

    def on_train_epoch_end(self):
        self.history['train_loss'].append(self.trainer.callback_metrics['train_loss'].item())
        self.history['train_acc'].append(self.trainer.callback_metrics['train_acc'].item())

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.history['val_acc'].append(self.trainer.callback_metrics['val_acc'].item())
            self.history['val_loss'].append(self.trainer.callback_metrics['val_loss'].item())


    def configure_optimizers(self):
        # Оптимизатор будет обновлять только размороженные веса
        return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)

# ==================================================

CHECKPOINTS_KR = Path('checkpoints')

# ==================================================

NUM_CLASSES = dm.num_classes
MAX_EPOCHS = 20
BATCH_SIZE = 32

filename = f'VGG16-{time.localtime()[2]}.{time.localtime()[1]}-{time.localtime()[3]}.{time.localtime()[4]}-'


checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINTS_KR,
    filename=filename+"{epoch:02d}-{val_acc:.4f}",
    save_top_k=1,
    monitor="val_acc",
    mode="max"
)

model_partial_ft = PartialFineTuneVGG(num_classes=NUM_CLASSES,learning_rate= 1e-4*3)
dm = DataModule(data_path=TRAIN_PATH/'train', zip_path=zip_path, batch_size=BATCH_SIZE)

dm.prepare_data()
dm.setup()

early_stop_callback = EarlyStopping(monitor='val_acc', patience=7, verbose=True, mode='max')
trainer_partial_ft = pl.Trainer(
    max_epochs=MAX_EPOCHS, 
    accelerator="auto", 
    callbacks=[early_stop_callback, checkpoint_callback] # TQDMProgressBar(refresh_rate=10), 
)


# ==================================================

trainer_partial_ft.fit(model_partial_ft, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.test_dataloader())

# ==================================================

from tqdm.auto import tqdm


cur = checkpoint_callback.best_model_path.split('\\')[-1]

loaded_m = PartialFineTuneVGG.load_from_checkpoint(CHECKPOINTS_KR/cur)

test_transform = T.Compose([
    T.Resize((300, 300)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=dm.mean, std=dm.std)
])
test_dataset = ImageFolder(root=TEST_PATH, transform=test_transform)


eral = DataLoader(test_dataset, num_workers=7, persistent_workers=True, shuffle=False)



# 1. Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Переносим модель на устройство и переводим в eval
loaded_m.to(device)
loaded_m.eval()

collect = []

# 3. Контекстный менеджер для отключения градиентов
# Это отключает построение графа вычислений (экономит RAM/VRAM)
with torch.inference_mode():
    # tqdm добавляет красивую полоску прогресса
    for i in tqdm(eral, desc="Running Inference"):
        x, _ = i # Игнорируем labels, если они не нужны для метрик
        
        # 4. Явно переносим данные на GPU
        # [Batch, Channels, Height, Width]
        x = x.to(device)
        
        # Forward pass
        # res: [Batch, Num_Classes] (обычно логиты)
        logits = loaded_m(x)
        
        # 5. Обработка результатов
        # Обычно нам нужны вероятности (softmax) или классы (argmax)
        # probabilities = torch.softmax(logits, dim=1)
        # preds = torch.argmax(probabilities, dim=1)
        
        # Важно: переносим обратно на CPU, чтобы не забить память GPU списком
        collect.append(logits.cpu())

final_predictions = torch.cat(collect, dim=0)

files = list(map(lambda x: x[0].split('\\')[-1], test_dataset.imgs))

ans = list(map(lambda x:dict(list(zip(range(dm.num_classes), dm.class_names))).get(x,None) , F.softmax(final_predictions, dim=1).argmax(dim=1).tolist()))

pd.DataFrame(pd.DataFrame([files, ans],index=['filename', 'class']).T).to_csv(f'submissions/{cur}.csv',index=False)

# ==================================================

def run_full_evaluation(model, loader, logger, history, classes, config):
    print("Running Full Evaluation...")
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    # 1. Inference
    with torch.inference_mode():
        # tqdm добавляет красивую полоску прогресса
        for i in tqdm(loader, desc="Running Inference"):
            x, y = i
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
    fig.savefig(os.path.join(logger,"comprehensive_report.png"))
    plt.close(fig)

    # Save CSV
    pd.DataFrame(rep).T.to_csv(os.path.join(logger, "detailed_metrics.csv"))
    
    print('Evaluation finished!')
    
    Accuracy_on_given_train = 0 # TODO: Написать логику валидации на изначальном трейновом датасете
    
    return Accuracy_on_given_train
    
run_full_evaluation(loaded_m, dm.train_dataloader(), '.', model_partial_ft.history, dm.class_names,{'device':device})

# ==================================================

dotenv.load_dotenv()

try:
    GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or os.environ.get("YOUR_USERNAME")
    if not GITHUB_USERNAME:
        raise ValueError("Не найден GITHUB_USERNAME")
        
    GITHUB_TOKEN = os.environ['GITHUB_DLTESTWORK_TOKEN']
    GITHUB_REPO_NAME = "kvdep/DL-TESTWORK"
    GIT_REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_REPO_NAME}.git"
    
except Exception as e:
    print(f"❌ Ошибка настройки: {e}")
    raise e

if 'cur' not in locals():
    print("⚠️ Переменная 'cur' не найдена. Использую заглушку.")
    sys.exit()

# 2. Подготовка путей
launch_dir_name = cur.split('.ckpt')[0]
launch_dir_name = 'VGG16_more_pics' 
launch_dir = os.path.join("dumplings", launch_dir_name)
os.makedirs(launch_dir, exist_ok=True)


run_full_evaluation(loaded_m, dm.train_dataloader(), launch_dir, model_partial_ft.history, dm.class_names,{'device':device})

# Копирование CSV
src_csv = f'submissions/{cur}.csv'
dst_csv = os.path.join(launch_dir, f'{cur}.csv')
if os.path.exists(src_csv):
    shutil.copy(src_csv, dst_csv)

# 3. Сборка run_code.py
try:
    nb_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]
    current_nb = __vsc_ipynb_file__  # if 'vsc_ipynb_file' in globals() else (nb_files[0] if nb_files else None)
    
    if current_nb:
        with open(current_nb, encoding='utf-8') as f:
            j = json.load(f)
            collect_cells = ""
            for cell in j['cells'][:-1]:
                if cell['cell_type'] == 'code':
                    collect_cells += "".join(cell['source']) + "\n\n# " + "="*50 + "\n\n"
        with open(os.path.join(launch_dir, 'run_code.py'), 'w+', encoding='utf-8') as f:
            f.write(collect_cells)
except Exception as e:
    print(f"⚠️ Ошибка сборки скрипта: {e}")


# ==================================================

