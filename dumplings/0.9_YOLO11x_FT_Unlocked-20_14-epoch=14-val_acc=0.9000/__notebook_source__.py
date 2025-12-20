


get_ipython().getoutput("pip install ultralytics")



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
from pytorch_lightning.loggers import MLFlowLogger, CSVLogger
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
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             auc, precision_recall_curve, average_precision_score, accuracy_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns
# import mlflow
from collections import defaultdict
from ultralytics import YOLO


torch.set_float32_matmul_precision('high')





import os
from pathlib import Path
from typing import List, Dict, Tuple
import imagehash
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

class DatasetCleaner:
    def __init__(self, root_dir: str, hash_size: int = 8, threshold: int = 0):
        self.root_dir = Path(root_dir)
        self.hash_size = hash_size
        self.threshold = threshold
        # Добавил .webp на всякий случай, в ML часто встречается
        self.extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def _compute_hash_and_info(self, file_path: Path) -> Tuple[str, Path, int]:
        try:
            with Image.open(file_path) as img:
                img_hash = imagehash.phash(img, hash_size=self.hash_size)
                width, height = img.size
                return str(img_hash), file_path, width * height
        except Exception:
            return None, file_path, -1

    def find_duplicates(self) -> Dict[str, List[Tuple[Path, int]]]:
        all_files = [p for p in self.root_dir.rglob('*') if p.suffix.lower() in self.extensions]
        print(f"🔍 Сканирую {len(all_files)} изображений (Single Thread Mode)...")

        results = []
        # --- ИЗМЕНЕНИЕ: Обычный цикл вместо Pool ---
        for f in tqdm(all_files, desc="Processing"):
            res = self._compute_hash_and_info(f)
            results.append(res)
        # -------------------------------------------

        hashes_dict = {}
        for h_str, path, area in results:
            if h_str:
                hashes_dict.setdefault(h_str, []).append((path, area))

        # Оставляем только те ключи, где больше 1 картинки
        return {k: v for k, v in hashes_dict.items() if len(v) > 1}

    def inspect_duplicates(self, num_samples: int = 5):
        duplicates = self.find_duplicates()
        
        if not duplicates:
            print("✅ Дубликатов нет.")
            return

        print(f"⚠️ Найдено групп дубликатов: {len(duplicates)}")
        print(f"👀 Показываю первые {num_samples} примеров...\n")

        sample_keys = list(duplicates.keys())[:num_samples]

        for h_str in sample_keys:
            file_list = duplicates[h_str]
            # Сортировка: Самое большое разрешение -> index 0 (Оставляем)
            file_list.sort(key=lambda x: x[1], reverse=True)
            
            keep_file, keep_area = file_list[0]
            del_file, del_area = file_list[1] 

            self._plot_comparison(keep_file, keep_area, del_file, del_area)

    def _plot_comparison(self, keep_path, keep_area, del_path, del_area):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            img_keep = Image.open(keep_path)
            axes[0].imshow(img_keep)
            axes[0].set_title(f"✅ KEEP\n{keep_path.name}\n{keep_area} px", color='green', fontweight='bold')
            axes[0].axis('off')

            img_del = Image.open(del_path)
            axes[1].imshow(img_del)
            axes[1].set_title(f"❌ DELETE\n{del_path.name}\n{del_area} px", color='red', fontweight='bold')
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()
            
            img_keep.close()
            img_del.close()
            
        except Exception as e:
            print(f"Ошибка отображения: {e}")
            
    def delete_duplicates(self):
        """
        РЕАЛЬНОЕ УДАЛЕНИЕ. Запускать только после проверки!
        """
        duplicates = self.find_duplicates()
        deleted_count = 0
        
        for file_list in tqdm(duplicates.values(), desc="Deleting"):
            file_list.sort(key=lambda x: x[1], reverse=True)
            # Все кроме первого (нулевого индекса) удаляем
            for rm_file, _ in file_list[1:]:
                try:
                    os.remove(rm_file)
                    deleted_count += 1
                except OSError as e:
                    print(f"Error: {e}")
        
        print(f"🔥 УДАЛЕНО файлов: {deleted_count}")
        
# for i in list(os.walk(r'/kaggle/input/dumplings/train'))[0][1]:
#     cleaner = DatasetCleaner(root_dir=r'C:\Projects\FU\Course_3\DL\data\yummi-classification-fu25\train'+ '\\'+ i)
#     cleaner.delete_duplicates()

# import os
# import shutil
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# from tqdm.auto import tqdm

# # ================= КОНФИГУРАЦИЯ =================
# SOURCE_DIR = TRAIN_PATH  # Где сейчас лежат папки gyoza, manti...
# CLEAN_DIR =  DATA_FOLDER / 'clean' # Куда сохранять чистые
# TRASH_DIR = DATA_FOLDER / 'trash_bin'     # Куда кидать мусор (для проверки)

# # Насколько модель должна быть уверена, что это НЕ еда, чтобы выкинуть (0.5 = 50%)
# TRASH_THRESHOLD = 0.6 

# # Текстовые промпты для CLIP
# # Мы сравниваем "food" против всего остального
# POS_PROMPT = "a photo of food, dumplings, meat, or dough"
# NEG_PROMPTS = [
#     "a photo of a person", 
#     "a photo of an empty plate", 
#     "a photo of a cat or dog",
#     "a photo of text or document",
#     "a blurred image",
#     "a photo of random object",
#     "noise"
# ]
# ALL_LABELS = [POS_PROMPT] + NEG_PROMPTS
# # ===============================================

# def clean_data():
#     # 1. Загружаем CLIP (Маленькая и быстрая версия)
#     print("🤖 Loading CLIP model...")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     # Создаем папки
#     if os.path.exists(CLEAN_DIR): shutil.rmtree(CLEAN_DIR)
#     if os.path.exists(TRASH_DIR): shutil.rmtree(TRASH_DIR)
    
#     os.makedirs(CLEAN_DIR, exist_ok=True)
#     os.makedirs(TRASH_DIR, exist_ok=True)

#     # 2. Сканируем файлы
#     file_list = []
#     for root, dirs, files in os.walk(SOURCE_DIR):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 file_list.append(os.path.join(root, file))

#     print(f"🔍 Found {len(file_list)} images. Starting scanning...")

#     # 3. Процесс фильтрации
#     # Обрабатываем по одной (можно батчами для ускорения, но так надежнее для скрипта)
#     kept_count = 0
#     trash_count = 0

#     with torch.no_grad():
#         for file_path in tqdm(file_list):
#             try:
#                 image = Image.open(file_path).convert("RGB")
                
#                 # Готовим данные для CLIP
#                 inputs = processor(
#                     text=ALL_LABELS, 
#                     images=image, 
#                     return_tensors="pt", 
#                     padding=True
#                 ).to(device)

#                 # Прогоняем через модель
#                 outputs = model(**inputs)
                
#                 # Считаем вероятности (Softmax)
#                 logits_per_image = outputs.logits_per_image # similarity score
#                 probs = logits_per_image.softmax(dim=1) # [1, len(ALL_LABELS)]

#                 # Индекс 0 - это POS_PROMPT ("food"), остальные - NEG
#                 prob_food = probs[0, 0].item()
#                 prob_trash = 1.0 - prob_food # Сумма вероятностей мусора

#                 # Определяем папку назначения
#                 # Сохраняем структуру классов (gyoza/img.jpg)
#                 rel_path = os.path.relpath(file_path, SOURCE_DIR)
                
#                 if prob_trash > TRASH_THRESHOLD:
#                     # Это мусор
#                     dest_path = os.path.join(TRASH_DIR, rel_path)
#                     trash_count += 1
#                 else:
#                     # Это еда
#                     dest_path = os.path.join(CLEAN_DIR, rel_path)
#                     kept_count += 1

#                 # Копируем
#                 os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#                 shutil.copy2(file_path, dest_path)

#             except Exception as e:
#                 print(f"Error processing {file_path}: {e}")

#     print(f"\n✅ Done!")
#     print(f"🥗 Kept (Clean): {kept_count}")
#     print(f"🗑️ Removed (Trash): {trash_count}")
#     print(f"Trash files are in {TRASH_DIR} - check them visually!")

# # Запуск
# # clean_data()


DATA_FOLDER = Path('/kaggle/input/dumplings')
TRUE_TEST_PATH = DATA_FOLDER / 'test'
TRAIN_PATH = DATA_FOLDER / 'clean'
VAL_PATH = DATA_FOLDER / "val"
CHECKPOINTS_KR = Path('/kaggle/working/dumplings_yolo')


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path: str,test_path:str = None, batch_size: int = 64, num_workers: int = 7, random_state = None):
        super().__init__()
        self.data_path = data_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.transform = None
        self.train_dataset = None
        self.test_dataset = None
        self.class_names = None
        self.num_classes = 0

        self.num_workers = num_workers
        self.random_state = random_state
        
    def _clean_dataset(self, folder_path: Path):
        """
        Проходит по папке, пытается открыть каждое изображение.
        Если изображение битое — удаляет его.
        """
        if not folder_path.exists():
            return

        print(f"🔍 Проверка целостности изображений в {folder_path}...")
        bad_files = 0
        total_files = 0
        
        # Рекурсивно проходим по всем файлам
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                total_files += 1
                try:
                    with Image.open(file_path) as img:
                        # Важно: просто open() ленивый, он читает только заголовок.
                        # verify() проверяет целостность данных, а convert/load загружает пиксели.
                        img.verify() 
                except (UnidentifiedImageError, OSError, IndexError) as e:
                    print(f"❌ Удален битый файл: {file_path} ({e})")
                    try:
                        os.remove(file_path)
                        bad_files += 1
                    except PermissionError:
                        print(f"⚠️ Не удалось удалить {file_path}, проверьте права доступа.")
        
        print(f"✅ Проверка завершена. Проверено: {total_files}. Удалено битых: {bad_files}.\n")

    def setup(self, stage: str = None):
        self._clean_dataset(self.data_path)
        # 1. Предварительная трансформация для расчета статистики
        # pre_transform = T.Compose([
        #     T.Resize((300, 300)),
        #     T.ToImage(),  
        #     T.ToDtype(torch.float32, scale=True) # Теперь работает с тензором
        # ])
        
        # full_dataset_for_stats = ImageFolder(root=self.data_path, transform=pre_transform)
        
        # # 2. Разбиение на train/test
        # train_size = int(0.8 * len(full_dataset_for_stats))
        # test_size = len(full_dataset_for_stats) - train_size
        # generator = torch.Generator().manual_seed(42)
        # train_subset_for_stats, _ = random_split(full_dataset_for_stats, [train_size, test_size], generator=generator)

        # 3. Расчет статистики по обучающей выборке
        # print("\nРасчет статистики для нормализации...")
        # loader = DataLoader(full_dataset_for_stats, batch_size=self.batch_size,)
        
        # mean = torch.zeros(3)
        # std = torch.zeros(3)
        # nb_samples = 0
        # for images, _ in loader:
        #     batch_samples = images.size(0)
        #     images = images.view(batch_samples, images.size(1), -1)
        #     mean += images.mean(2).sum(0)
        #     std += images.std(2).sum(0)
        #     nb_samples += batch_samples
        
        # mean /= nb_samples
        # std /= nb_samples
        # print(f"Рассчитанное среднее (mean): {mean}")
        # print(f"Рассчитанное стандартное отклонение (std): {std}")

        self.train_transform1 = T.Compose([
            T.Resize((380, 380)),
            T.RandomHorizontalFlip(p=0.5), # Случайное горизонтальное отражение
            T.RandomRotation(degrees=15), # Случайный поворот на 15 градусов
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Случайное изменение яркости/контраста
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # T.Normalize(mean=mean, std=std)
        ])
        
        self.train_transform2 = T.Compose([
            # RandomResizedCrop заставляет модель смотреть на детали, а не только на общую форму
            T.RandomResizedCrop(size=(380, 380), scale=(0.8, 1.0)), 
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # T.Normalize(mean=mean, std=std)
        ])

        # Трансформации для ТЕСТОВОГО набора (без аугментации!)
        self.test_transform = T.Compose([
            T.Resize((380, 380)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # T.Normalize(mean=mean, std=std)
        ])
        
        
        # self.std = std
        # self.mean = mean
        # 5. Создание финальных датасетов с разными трансформациями
        dataset_train1 = ImageFolder(root=self.data_path, transform=self.train_transform1)
        dataset_train2 = ImageFolder(root=self.data_path, transform=self.train_transform2)
        dataset_train3 = ImageFolder(root=self.data_path, transform=self.test_transform)
        dataset_test = ImageFolder(root=self.test_path, transform=self.test_transform)
        
        self.class_names = dataset_test.classes
        self.num_classes = len(self.class_names)
        self.transforms = {
            'test':self.test_transform,
            'train1':self.train_transform1,
            'train2':self.train_transform2
        }
        
        
        # Важно! Мы должны разделить один и тот же "исходный" датасет
        # random_split не может работать с двумя разными объектами датасетов
        # full_dataset_final = ImageFolder(root=self.data_path) # Без трансформаций
        # train_size = int(0.8 * len(full_dataset_final))
        # test_size = len(full_dataset_final) - train_size
        # self.dataset = full_dataset_final
        
        # generator = torch.Generator().manual_seed(42)
        # train_indices, test_indices = random_split(range(len(full_dataset_final)), [train_size, test_size], generator=generator)
        # if self.random_state==None:
        #     rs = np.random.randint(0,1000)
        # else:
        #     rs = self.random_state
        # train_indices, test_indices = train_test_split(range(len(full_dataset_final)), stratify=full_dataset_final.targets, test_size=0.2, random_state=rs)
        # print(rs)

        
        # Теперь применяем нужные трансформации к нужным подвыборкам
        # self.train_dataset = torch.utils.data.Subset(dataset_train, train_indices.indices)
        # self.test_dataset = torch.utils.data.Subset(dataset_test, test_indices.indices)
        # subset1 = Subset(dataset_train1, train_indices)
        # subset2= Subset(dataset_train2, train_indices)
        # subset3 = Subset(dataset_train3, train_indices)
        
        # self.train_dataset =  ConcatDataset([subset1, subset2, subset3])
        self.train_dataset =  ConcatDataset([dataset_train1, dataset_train2, dataset_train3])
        # self.train_dataset =  ConcatDataset([dataset_train1, dataset_train2])


        # self.test_dataset = Subset(dataset_test, test_indices)
        self.test_dataset = dataset_test
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self, transform='test'):
        dataset_test = ImageFolder(root=self.test_path, transform=self.transforms[transform])
        return DataLoader(dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self, transform='test'):
        return self.test_dataloader(transform)


dm = DataModule(data_path=TRAIN_PATH,test_path=VAL_PATH, batch_size=32)

dm.setup()


print(f"Размер обучающей выборки: {len(dm.train_dataset)}")
print(f"Размер тестовой выборки: {len(dm.test_dataset)}")

image_sample, _ = dm.train_dataset[0]


print(f"\nОбщее количество картинок в датасете: {len(dm.train_dataset) + len(dm.test_dataset)}")
print(f"Размер одной картинки (тензора): {image_sample.shape}")
print(f"Количество уникальных классов: {dm.num_classes}")
print("Названия классов:", dm.class_names)





class YOLOv11Classifier(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3, freeze_backbone: bool = True): # LR 1e-3
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Загрузка
        print("🏗️ Loading YOLOv11x-cls pre-trained weights...")
        yolo_wrapper = YOLO('yolo11x-cls.pt') 
        self.model = yolo_wrapper.model
        
        # 2. Модификация головы (как мы делали раньше)
        classification_head = self.model.model[-1]
        if hasattr(classification_head, 'linear') and isinstance(classification_head.linear, nn.Linear):
            in_features = classification_head.linear.in_features
            classification_head.linear = nn.Linear(in_features, num_classes)
        elif isinstance(classification_head, nn.Linear):
            in_features = classification_head.in_features
            self.model.model[-1] = nn.Linear(in_features, num_classes)
        else:
            # Fallback
            current_linear = list(classification_head.modules())[-1]
            if isinstance(current_linear, nn.Linear):
                classification_head.linear = nn.Linear(current_linear.in_features, num_classes)

        # 3. Заморозка
        if freeze_backbone:
            print("🔒 Backbone is FROZEN, but BatchNorm will adapt.")
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Размораживаем голову
            for param in self.model.model[-1].parameters():
                param.requires_grad = True
        else:
            print("🔓 Backbone is UNFROZEN (Full Fine-Tuning).")
            for param in self.model.parameters():
                param.requires_grad = True

        # Метрики
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_loss':[]}

    # ============================================================
    # 🔥 ГЛАВНОЕ ИЗМЕНЕНИЕ ДЛЯ ШАГА 3
    # ============================================================
    def train(self, mode=True):
        """
        Принудительно включаем обучение для BatchNorm, 
        даже если self.freeze_backbone = True
        """
        super().train(mode)
        
        # Если модель переводится в режим обучения (mode=True)
        # и мы хотим адаптировать статистики BN:
        if mode and self.hparams.freeze_backbone:
            for module in self.modules():
                # Ищем все слои BatchNorm
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train() # Разрешаем обновление running_mean / running_var
                    # Опционально: можно разморозить и веса BN (gamma/beta), раскомментировав ниже:
                    # for param in module.parameters():
                    #     param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    # ... (Остальные методы: training_step, validation_step и т.д. без изменений)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
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
        self.log('test_loss', loss)
        self.log('test_acc', self.val_acc(logits, y))

    def on_train_epoch_end(self):
        t_loss = self.trainer.callback_metrics.get('train_loss')
        t_acc = self.trainer.callback_metrics.get('train_acc')
        if t_loss is not None:
            self.history['train_loss'].append(t_loss.item())
            self.history['train_acc'].append(t_acc.item())

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            v_loss = self.trainer.callback_metrics.get('val_loss')
            v_acc = self.trainer.callback_metrics.get('val_acc')
            if v_loss is not None:
                self.history['val_loss'].append(v_loss.item())
                self.history['val_acc'].append(v_acc.item())

    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

MODEL = YOLOv11Classifier






# 1. Параметры (YOLO 'x' версия тяжелая, уменьшаем Batch Size если будет OOM Error)
BATCH_SIZE = 64 # Уменьшил с 32/64, так как модель X очень большая
LR = 1e-4     # Меньше LR для дообучения
MAX_EPOCHS = 15

# 2. Инициализация DataModule (предполагаем, что dm уже создан выше из твоего кода)
# Важно: YOLO v11 обучена на 224x224 или 640x640. Твои 300x300 или 380x380 подойдут.
# Но если памяти мало - уменьши размер картинки в DataModule.
dm = DataModule(data_path=TRAIN_PATH, test_path=VAL_PATH, batch_size=BATCH_SIZE, num_workers=12)
dm.setup()

# 3. Инициализация модели
# freeze_backbone=True для начала, чтобы не сломать веса при случайной инициализации новой головы
model_yolo = YOLOv11Classifier(num_classes=dm.num_classes, learning_rate=LR, freeze_backbone=True)


print("\n🔍 ПРОВЕРКА ОБУЧАЕМЫХ ПАРАМЕТРОВ:")
trainable_params = []
for name, param in model_yolo.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)
        print(f" ✅ Trainable: {name} | Shape: {param.shape}")

if not trainable_params:
    print("❌ ОШИБКА: Нет обучаемых параметров! Модель заморожена полностью.")
elif len(trainable_params) < 2: # Обычно weight и bias
    print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Обучается подозрительно мало слоев.")
else:
    print(f"🆗 Всего обучаемых тензоров: {len(trainable_params)}")
    
# Проверка последнего слоя
last_layer_check = False
for name in trainable_params:
    if "model.2" in name or "linear" in name or "fc" in name or f"{len(model_yolo.model.model)-1}" in name:
        last_layer_check = True

if not last_layer_check:
    print("💀 КРИТИЧЕСКАЯ ОШИБКА: Голова классификации не в списке trainable!")


def visualize_batch(dataloader):
    # Берем один батч
    images, labels = next(iter(dataloader))
    
    print(f"Tensor Range: Min={images.min():.3f}, Max={images.max():.3f}")
    print(f"Tensor Shape: {images.shape}")
    print(f"Labels: {labels}")
    
    # Визуализация
    plt.figure(figsize=(16, 8))
    for i in range(min(8, len(images))):
        ax = plt.subplot(2, 4, i + 1)
        # Permute: [C, H, W] -> [H, W, C]
        img = images[i].permute(1, 2, 0).numpy()
        
        # Если данные [0, 1], matplotlib покажет нормально
        # Если данные нормализованы (есть отрицательные), надо раз-нормализовать для просмотра
        if img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min())
            
        plt.imshow(img)
        plt.title(f"Class: {dm.class_names[labels[i]]}")
        plt.axis("off")
    plt.show()

# Запускаем проверку
visualize_batch(dm.train_dataloader())




# 4. Callbacks и Logger
filename_yolo = f'YOLO11x_CLS-{time.localtime()[2]}.{time.localtime()[1]}-{time.localtime()[3]}.{time.localtime()[4]}'

checkpoint_callback_yolo = ModelCheckpoint(
    dirpath=CHECKPOINTS_KR,
    filename=filename_yolo+"-{epoch:02d}-{val_acc:.4f}",
    save_top_k=1,
    monitor="val_acc",
    mode="max"
)

trainer_yolo = pl.Trainer(
    max_epochs=8, 
    accelerator="auto", 
    callbacks=[checkpoint_callback_yolo],
    logger=[CSVLogger(save_dir='/kaggle/working/dumplings_yolo', name=filename_yolo)]
)

trainer_yolo.fit(model_yolo, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.test_dataloader())


# 1. Берем последний или лучший чекпоинт
# Если checkpoint_callback_yolo не определен, можно взять путь руками из папки checkpoints
best_model_path = checkpoint_callback_yolo.best_model_path
print(f"💎 Загружаем веса из: {best_model_path}")

# 2. Загружаем модель для Fine-Tuning
# ВАЖНО: 
# - freeze_backbone=False (Размораживаем всё)
# - learning_rate=1e-5 (Уменьшаем в 100 раз! Было 1e-3. Иначе сломаем веса)
fine_tune_model = YOLOv11Classifier.load_from_checkpoint(
    best_model_path,
    num_classes=dm.num_classes,
    learning_rate=1e-5,       # <--- ОЧЕНЬ ВАЖНО
    freeze_backbone=False     # <--- СВОБОДА
)

# 3. Настраиваем Trainer
# ВАЖНО: Уменьши batch_size в DataModule, если будет OOM (Out of Memory).
# Размороженная модель ест больше памяти.
dm_ft = DataModule(data_path=TRAIN_PATH, test_path=VAL_PATH, batch_size=16) # Поставь 16 или 32
dm_ft.setup()

filename_ft = f'YOLO11x_FT_Unlocked-{time.localtime()[2]}_{time.localtime()[3]}'

checkpoint_callback_ft = ModelCheckpoint(
    dirpath=CHECKPOINTS_KR,
    filename=filename_ft+"-{epoch:02d}-{val_acc:.4f}",
    save_top_k=5,
    monitor="val_acc",
    mode="max"
)

# EarlyStopping, чтобы не ждать зря, если не пойдет
early_stop = EarlyStopping(monitor="val_acc", patience=5, mode="max")

trainer_ft = pl.Trainer(
    max_epochs=15,             # Дадим ей время перестроиться
    accelerator="auto",
    devices=1,
    callbacks=[checkpoint_callback_ft, early_stop],
    logger=[CSVLogger(save_dir='/kaggle/working/dumplings_yolo', name=filename_yolo)]
)

print("🚀 ЗАПУСК ПОЛНОЙ РАЗМОРОЗКИ...")
trainer_ft.fit(fine_tune_model, train_dataloaders=dm_ft.train_dataloader(), val_dataloaders=dm_ft.test_dataloader())


trainer_ft.fit(fine_tune_model, train_dataloaders=dm_ft.test_dataloader(), val_dataloaders=dm_ft.test_dataloader())





from tqdm.auto import tqdm


cur = checkpoint_callback_ft.best_model_path.split('/')[-1]

loaded_m = MODEL.load_from_checkpoint(CHECKPOINTS_KR/cur)

test_transform = T.Compose([
    T.Resize((380, 380)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    # T.Normalize(mean=dm.mean, std=dm.std)
])
test_dataset = ImageFolder(root=TRUE_TEST_PATH, transform=test_transform)


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

files = list(map(lambda x: x[0].split('/')[-1], test_dataset.imgs))


ans = list(map(lambda x:dict(list(zip(range(dm.num_classes), dm.class_names))).get(x,None) , F.softmax(final_predictions, dim=1).argmax(dim=1).tolist()))
os.makedirs('/kaggle/working/subs',exist_ok=True)
pd.DataFrame(pd.DataFrame([files, ans],index=['filename', 'class']).T).to_csv(f'/kaggle/working/submissions.{cur}.csv',index=False)
print(f'/kaggle/working/submissions.{cur}.csv')


def run_full_evaluation(model, loader, logger_path, history, classes, config, tta_transforms=None, show=False):
    """
    Args:
        tta_transforms (list): Список функций для аугментации (принимают tensor, возвращают tensor).
    """
    print(f"Running Full Evaluation with TTA (x{len(tta_transforms) if tta_transforms else 1})...")
    model.eval()
    
    # Если трансформации не переданы, используем только оригинал
    if tta_transforms is None:
        tta_transforms = [lambda x: x]

    y_true, y_pred, y_prob = [], [], []

    # 1. Inference Loop
    with torch.inference_mode():
        for i in tqdm(loader, desc="Inference + TTA"):
            x, y = i
            x = x.to(config['device'])
            
            # --- TTA LOGIC START ---
            batch_probs = []
            
            # Прогоняем каждый вариант аугментации
            for t in tta_transforms:
                # Применяем трансформацию к батчу
                x_aug = t(x) 
                
                # Получаем логиты
                out = model(x_aug)
                
                # Считаем вероятности (Softmax)
                # Важно усреднять именно вероятности, а не логиты!
                prob = torch.nn.functional.softmax(out, dim=1)
                batch_probs.append(prob)
            
            # Усредняем вероятности по всем аугментациям (Stack -> Mean)
            # shape: [Num_Augs, Batch_Size, Num_Classes] -> [Batch_Size, Num_Classes]
            avg_prob = torch.stack(batch_probs).mean(dim=0)
            # --- TTA LOGIC END ---

            y_true.extend(y.cpu().numpy())
            # Предсказание берем по максимальной УСРЕДНЕННОЙ вероятности
            y_pred.extend(torch.max(avg_prob, 1)[1].cpu().numpy())
            y_prob.extend(avg_prob.cpu().numpy())

    # Преобразуем в numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    n_classes = len(classes)

    # 2. Visualization (Остальной код построения графиков без изменений)
    #plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3)

    # A. Training Dynamics (Loss & Acc)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    # Проверка, есть ли история (для загруженных чекпоинтов может быть пустой)
    if history and 'train_loss' in history and len(history['train_loss']) > 0:
        epochs = range(1, len(history['train_loss']) + 1)
        l1 = ax1.plot(epochs, history['train_loss'], 'r-o', lw=2, label='Train Loss')
        # Проверка длины массивов (иногда валидация чаще/реже)
        val_len = len(history['val_acc'])
        l2 = ax1_twin.plot(range(1, val_len + 1), history['val_acc'], 'c-o', lw=2, label='Val Acc')
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
    else:
        ax1.text(0.5, 0.5, "No History Available", ha='center')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='r')
    ax1_twin.set_ylabel('Accuracy', color='c')
    ax1.set_title('Training Dynamics', fontsize=14, fontweight='bold', color='black')
    ax1.grid(True, alpha=0.1)

    # B. Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax2,
                xticklabels=classes, yticklabels=classes, cbar=False)
    ax2.set_title(f'Confusion Matrix (TTA x{len(tta_transforms)})', fontsize=14, fontweight='bold', color='black')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # C. Per-Class Metrics Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_rep = pd.DataFrame(rep).T.iloc[:-3, :3] 
    sns.heatmap(df_rep, annot=True, fmt='.3f', cmap='viridis', ax=ax3, cbar=False)
    ax3.set_title('Class-wise Metrics', fontsize=14, fontweight='bold', color='black')

    # D. ROC Curves
    ax4 = fig.add_subplot(gs[1, 0])
    y_bin = label_binarize(y_true, classes=range(n_classes))
    for i in range(n_classes):
        if n_classes == 2: # Binary case fix
             fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        else:
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

    # F. Confidence Histogram
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
    
    # Пытаемся достать лучший val_acc из истории, если она есть
    best_val_txt = "N/A"
    if history and 'val_acc' in history and len(history['val_acc']) > 0:
        best_val_txt = f"{max(history['val_acc']):.4f}"

    txt = f"Global Accuracy (with TTA): {acc:.2%}\n"
    txt += f"Best Training Val Acc: {best_val_txt}\n"
    txt += f"Total Samples: {len(y_true)}"
    ax7.text(0.5, 0.5, txt, ha='center', va='center', fontsize=20, color='black')

    # Save
    fig.savefig(os.path.join(logger_path, "comprehensive_report_tta.png"))
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Save CSV
    pd.DataFrame(rep).T.to_csv(os.path.join(logger_path, "detailed_metrics_tta.csv"))

    
    # pd.DataFrame(pd.DataFrame([files, ans],index=['filename', 'class']).T).to_csv(f'/kaggle/working/submissions/{cur}.csv',index=False)


    
    print(f'Evaluation finished! Accuracy: {acc:.4f}')
    return acc


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pandas as pd
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as T # Используем v2 для скорости

# 1. Загрузка весов
cur = checkpoint_callback_ft.best_model_path.split('/')[-1]

print(f"🔮 Generating submission using: {cur}")
loaded_m = YOLOv11Classifier.load_from_checkpoint(checkpoint_callback_ft.best_model_path)
# loaded_m = SotaDumplingModel.load_from_checkpoint(...) # Если используешь класс SOTA

# 2. Настройка датасета (Без нормализации для YOLO!)
test_transform = T.Compose([
    T.Resize((380, 380)), # Убедись, что размер совпадает с обучением
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])

# Важно: ImageFolder требует структуру папок /root/class/img.png.
# Если у тебя в TRUE_TEST_PATH просто куча картинок без папок, ImageFolder может не сработать.
# Обычно на Kaggle тест лежит в папке 'test_images', поэтому указываем путь на уровень выше или используем кастомный Dataset.
# Предполагаем, что TRUE_TEST_PATH указывает на корень, где лежат папки (или одна папка 'unknown').
test_dataset = ImageFolder(root=TRUE_TEST_PATH, transform=test_transform)

# DataLoader
eral = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False, pin_memory=True)

# 3. Подготовка TTA (Test Time Augmentation)
# Функции принимают тензор на GPU и возвращают тензор
tta_transforms = [
    lambda x: x,                      # 1. Оригинал
    lambda x: TF.hflip(x),            # 2. Отражение по горизонтали
]

# 4. Inference Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_m.to(device)
loaded_m.eval()

collect_probs = []

print(f"🚀 Starting TTA Inference (x{len(tta_transforms)} augmentations)...")

with torch.inference_mode():
    for i in tqdm(eral, desc="Predicting"):
        x, _ = i 
        x = x.to(device)
        
        batch_tta_probs = []
        
        # --- TTA LOOP ---
        for transformer in tta_transforms:
            # А. Аугментация батча на лету (на GPU это быстро)
            x_aug = transformer(x)
            
            # Б. Прогон через модель
            logits = loaded_m(x_aug)
            
            # В. Получаем вероятности СРАЗУ (Softmax)
            probs = F.softmax(logits, dim=1)
            batch_tta_probs.append(probs)
        
        # Г. Усредняем вероятности всех аугментаций
        # Stack [5, Batch, 4] -> Mean(dim=0) -> [Batch, 4]
        avg_prob = torch.stack(batch_tta_probs).mean(dim=0)
        
        # Д. Сохраняем результат (на CPU, чтобы не забить память GPU)
        collect_probs.append(avg_prob.cpu())

# 5. Сборка результатов
final_probs = torch.cat(collect_probs, dim=0)

# Получаем имена файлов (ImageFolder возвращает полные пути, берем только имя файла)
files = [os.path.basename(x[0]) for x in test_dataset.imgs]

# Получаем предсказания (Argmax по усредненным вероятностям)
# F.softmax тут уже НЕ НУЖЕН, так как final_probs уже вероятности
predicted_indices = final_probs.argmax(dim=1).tolist()

# Маппинг индексов в названия классов
idx_to_class = {i: name for i, name in enumerate(dm.class_names)}
ans = [idx_to_class.get(idx) for idx in predicted_indices]

# 6. Сохранение
os.makedirs('/kaggle/working/submissions', exist_ok=True)
sub_name = f'/kaggle/working/submissions/{cur}_TTAx{len(tta_transforms)}.csv'

df = pd.DataFrame({'filename': files, 'class': ans})
df.to_csv(sub_name, index=False)

print(f"✅ Submission saved to: {sub_name}")
print(df.head())


# 1. Задаем трансформации
tta_list = [
    lambda x: x,              # Оригинал
    lambda x: TF.hflip(x),    # Отражение по горизонтали
]

# 2. Запускаем
# Обрати внимание: model_yolo нужно перевести на .to(device) внутри функции или до нее. 
# В моем коде функции есть x.to(config['device']), но сама модель должна быть там же.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_m.to(device) # Убедись, что модель на GPU

acc = run_full_evaluation(
    model=loaded_m, 
    loader=dm.test_dataloader(), 
    logger_path='.', 
    history=fine_tune_model.history, # Или history из YOLO 
    classes=dm.class_names,
    config={'device': device}, 
    tta_transforms=tta_list, # <--- Передаем список
    show=True
)


def run_full_evaluation(model, loader, logger, history, classes, config, show=False):
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
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Save CSV
    pd.DataFrame(rep).T.to_csv(os.path.join(logger, "detailed_metrics.csv"))
    
    print('Evaluation finished!')
    
    
    return acc
    
run_full_evaluation(loaded_m, dm.test_dataloader(), '.', fine_tune_model.history, dm.class_names,{'device':device}, True)


# 3. Сборка run_code.py\
import json
try:
    nb_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]
    current_nb =  r'/kaggle/working/.virtual_documents/__notebook_source__.ipynb'
    
    if current_nb:
        with open(current_nb, encoding='utf-8') as f:
            j = json.load(f)
            collect_cells = ""
            for cell in j['cells'][:-1]:
                if cell['cell_type'] == 'code':
                    collect_cells += "".join(cell['source']) + "\n\n# " + "="*50 + "\n\n"
        with open(os.path.join('.', 'run_code.py'), 'w+', encoding='utf-8') as f:
            f.write(collect_cells)
except Exception as e:
    print(f"⚠️ Ошибка сборки скрипта: {e}")


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()


dotenv.load_dotenv()

try:
    # GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or os.environ.get("YOUR_USERNAME")
    GITHUB_USERNAME = user_secrets.get_secret("GITHUB_USERNAME")
    if not GITHUB_USERNAME:
        raise ValueError("Не найден GITHUB_USERNAME")
        
    GITHUB_TOKEN = user_secrets.get_secret('GITHUB_DLTESTWORK_TOKEN')
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
launch_dir = os.path.join('/kaggle/working/dumplings', launch_dir_name)
os.makedirs(launch_dir, exist_ok=True)


acc = run_full_evaluation(loaded_m, dm.test_dataloader(), launch_dir, fine_tune_model.history, dm.class_names,{'device':device})

launch_dir_name = str(acc)+'_'+launch_dir_name
# Копирование CSV
src_csv = f'submissions/{cur}.csv'
dst_csv = os.path.join(launch_dir, f'{cur}.csv')
if os.path.exists(src_csv):
    shutil.copy(src_csv, dst_csv)
    
shutil.copytree(os.path.join('/kaggle/working/dumplings_yolo', filename_yolo, 'version_1'), launch_dir , dirs_exist_ok=True)

# 3. Сборка run_code.py
try:
    nb_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]
    current_nb =  '/kaggle/working/.virtual_documents/__notebook_source__.ipynb'
    
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

# 4. Git операции
temp_clone_dir = tempfile.mkdtemp()
print(f"Временная папка: {temp_clone_dir}")

def git_run(args, desc, ignore_error=False):
    # ДОБАВЛЕНО: encoding='utf-8', errors='replace'
    # errors='replace' заменит неизвестные символы на вопросики, чтобы точно не упасть
    res = subprocess.run(
        args, 
        cwd=temp_clone_dir, 
        capture_output=True, 
        encoding='utf-8', 
        errors='replace' 
    )
    
    if res.returncode != 0 and not ignore_error:
        print(f"❌ Ошибка на этапе {desc}:")
        print(res.stderr)
        raise RuntimeError(f"{desc} failed")
    return res

try:
    print(f"⏳ Клонирование (no-checkout)...")
    
    # 1. Клонируем, но НЕ трогаем файлы
    git_run([
        "git", "clone", 
        "--depth", "1", 
        "--no-checkout", 
        "-c", "http.postBuffer=524288000",
        GIT_REPO_URL, 
        temp_clone_dir
    ], "git clone")
    
    # 2. !!! МАГИЯ ДЛЯ WINDOWS !!!
    # Отключаем защиту NTFS, чтобы Git позволил существовать "плохим" именам в памяти (индексе)
    print("🔧 Отключение core.protectNTFS...")
    git_run(["git", "config", "core.protectNTFS", "false"], "config protectNTFS")
    
    # 3. Теперь reset должен сработать, так как мы разрешили "плохие" пути в индексе
    print("⏳ Восстановление индекса (git reset)...")
    git_run(["git", "reset"], "git reset")

    # 4. Копируем ВАШИ файлы
    dest_path = os.path.join(temp_clone_dir, "dumplings", launch_dir_name)
    print(f"📂 Копирование файлов в {dest_path}...")
    shutil.copytree(launch_dir, dest_path, dirs_exist_ok=True)
    
    # 5. Настройка автора
    git_run(['git', 'config', 'user.name', GITHUB_USERNAME], "config name")
    git_run(['git', 'config', 'user.email', f'{GITHUB_USERNAME}@users.noreply.github.com'], "config email")
    
    # 6. Добавляем папку
    # Используем относительный путь для add
    path_to_add = f"dumplings/{launch_dir_name}"
    print(f"➕ Добавление папки: {path_to_add}")
    
    # Важно: добавляем только нашу папку. 
    # Так как мы в режиме no-checkout, git status может показывать, что остальные файлы "удалены". 
    # Но git add <папка> застейджит только нашу папку, а commit зафиксирует это поверх текущего состояния.
    git_run(['git', 'add', path_to_add], "git add")
    
    # 7. Коммит и Пуш
    try:
        msg = f"{acc:.4f}"
    except:
        msg = f"Update {launch_dir_name}"
        
    print(f"🚀 Отправка: '{msg}'...")
    git_run(['git', 'commit', '-m', msg], "commit")
    git_run(['git', 'push'], "push")
    
    print("\n🎉 ПОБЕДА! Файлы отправлены.")

except Exception as e:
    print(f"\n❌ СКРИПТ УПАЛ: {e}")
finally:
    try:
        shutil.rmtree(temp_clone_dir, ignore_errors=True)
    except:
        pass
