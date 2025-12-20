
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
import mlflow
from collections import defaultdict


torch.set_float32_matmul_precision('high')

# ==================================================

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
# for i in list(os.walk(r'C:\Projects\FU\Course_3\DL\data\yummi-classification-fu25\train'))[0][1]:
#     cleaner = DatasetCleaner(root_dir=r'C:\Projects\FU\Course_3\DL\data\yummi-classification-fu25\train'+ '\\'+ i)
#     cleaner.delete_duplicates()

# ==================================================

DATA_FOLDER = Path('data') /'yummi-classification-fu25'
TRUE_TEST_PATH = DATA_FOLDER / 'test'
TRAIN_PATH = DATA_FOLDER / 'clean' # 'train
VAL_PATH = DATA_FOLDER / "val"
CHECKPOINTS_KR = Path('checkpoints')


# ==================================================

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

# ==================================================

# TRAIN_PATH = CLEAN_DIR

# ==================================================

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
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5), # Случайное горизонтальное отражение
            T.RandomRotation(degrees=15), # Случайный поворот на 15 градусов
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Случайное изменение яркости/контраста
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # T.Normalize(mean=mean, std=std)
        ])
        
        self.train_transform2 = T.Compose([
            # RandomResizedCrop заставляет модель смотреть на детали, а не только на общую форму
            T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)), 
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # T.Normalize(mean=mean, std=std)
        ])

        # Трансформации для ТЕСТОВОГО набора (без аугментации!)
        self.test_transform = T.Compose([
            T.Resize((224, 224)),
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

# ==================================================

dm = DataModule(data_path=TRAIN_PATH,test_path=VAL_PATH, batch_size=32)

dm.setup()


print(f"Размер обучающей выборки: {len(dm.train_dataset)}")
print(f"Размер тестовой выборки: {len(dm.test_dataset)}")

image_sample, _ = dm.train_dataset[0]


print(f"\nОбщее количество картинок в датасете: {len(dm.train_dataset) + len(dm.test_dataset)}")
print(f"Размер одной картинки (тензора): {image_sample.shape}")
print(f"Количество уникальных классов: {dm.num_classes}")
print("Названия классов:", dm.class_names)

# ==================================================

# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# import torch.optim as optim
# from torchmetrics import Accuracy
# import numpy as np

# class SotaDumplingModel(pl.LightningModule):
#     def __init__(self, num_classes: int, model_name='convnext_base.fb_in22k_ft_in1k_384', learning_rate=1e-4):
#         """
#         SOTA Architecture wrapper.
#         Рекомендуемые модели (model_name):
#         - 'convnext_base.fb_in22k_ft_in1k_384' (Отличный баланс)
#         - 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k' (Трансформер)
#         - 'efficientnet_b5.ra2_in1k' (Если любишь ENet)
#         """
#         super().__init__()
#         self.save_hyperparameters()

#         # 1. Загрузка SOTA модели через timm
#         # pretrained=True грузит веса. num_classes меняет голову автоматически.
#         self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
#         # 2. Loss с Label Smoothing
#         # label_smoothing=0.1 говорит модели: "не будь уверена на 100%"
#         self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#         # Метрики
#         self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
#         self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

#         # Параметры для MixUp/CutMix
#         self.mixup_alpha = 0.2
#         self.cutmix_alpha = 1.0
#         self.mix_prob = 0.5 # Вероятность применения

#     def forward(self, x):
#         return self.model(x)

#     # === РЕАЛИЗАЦИЯ MIXUP / CUTMIX ВНУТРИ BATCH ===
#     def apply_mixup_cutmix(self, x, y):
#         # Решаем, применять ли вообще
#         if np.random.rand() > self.mix_prob:
#             return x, y, None, None, None

#         # Решаем: MixUp или CutMix?
#         use_cutmix = np.random.rand() > 0.5
        
#         # Генерируем перемешанные индексы
#         lam = 1.0
#         indices = torch.randperm(x.size(0)).to(x.device)
        
#         if use_cutmix:
#             # CutMix Logic
#             lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
#             bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
#             x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
#             # Корректируем лямбду под реальную площадь
#             lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
#         else:
#             # MixUp Logic
#             lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
#             x = lam * x + (1 - lam) * x[indices]
            
#         return x, y, indices, lam, "mixed"

#     def rand_bbox(self, size, lam):
#         W = size[2]
#         H = size[3]
#         cut_rat = np.sqrt(1. - lam)
#         cut_w = int(W * cut_rat)
#         cut_h = int(H * cut_rat)

#         # uniform
#         cx = np.random.randint(W)
#         cy = np.random.randint(H)

#         bbx1 = np.clip(cx - cut_w // 2, 0, W)
#         bby1 = np.clip(cy - cut_h // 2, 0, H)
#         bbx2 = np.clip(cx + cut_w // 2, 0, W)
#         bby2 = np.clip(cy + cut_h // 2, 0, H)

#         return bbx1, bby1, bbx2, bby2

#     def training_step(self, batch, batch_idx):
#         x, y = batch
        
#         # Применяем MixUp/CutMix
#         x_mixed, y_a, y_b, lam, status = self.apply_mixup_cutmix(x, y)
        
#         logits = self(x_mixed)
        
#         if status == "mixed":
#             # Loss считается как взвешенная сумма
#             loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
#         else:
#             loss = self.criterion(logits, y)

#         # Для Accuracy на train лучше считать по чистым меткам (хотя это не совсем честно при mixup)
#         # Но для мониторинга сойдет
#         preds = torch.argmax(logits, dim=1)
#         # self.train_acc(preds, y) # Можно отключить, т.к. при mixup метрики train шумные
        
#         self.log('train_loss', loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         # На валидации MixUp НЕ ПРИМЕНЯЕМ!
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         self.val_acc(logits, y)
#         self.log('val_loss', loss, prog_bar=True)
#         self.log('val_acc', self.val_acc, prog_bar=True)

#     def configure_optimizers(self):
#         # AdamW + CosineAnnealing
#         # Weight Decay важен для ConvNeXt (0.05 - стандарт)
#         optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.05)
        
#         # Cosine Schedule с разогревом (Warmup можно реализовать отдельно, но просто Cosine уже топ)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, 
#             T_max=self.trainer.max_epochs, 
#             eta_min=1e-6
#         )
        
#         return [optimizer], [scheduler]

# MODEL = SotaDumplingModel

# ==================================================

class PartialFineTuneVGG(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4): # Используем меньший learning rate
        super().__init__()
        self.save_hyperparameters()

        weights = models.VGG16_Weights.DEFAULT
        self.model = models.vgg16(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = True
            
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
    
MODEL = PartialFineTuneVGG

# ==================================================

def run_screenshot_script(url, output_path, width=1920, height=1480, sleep=5):
    """Запускает Playwright в подпроцессе для снятия скриншота."""
    script_code = """
import sys
import time
from playwright.sync_api import sync_playwright

def take_screenshot(url, output_path, width, height, sleep):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_viewport_size({"width": int(width), "height": int(height)})
            page.goto(url)
            time.sleep(int(sleep)) 
            page.screenshot(path=output_path, full_page=True)
            browser.close()
        print(f"Picture saved: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    take_screenshot(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
"""
    try:
        subprocess.run(
            [sys.executable, "-c", script_code, url, output_path, str(width), str(height), str(sleep)],
            capture_output=True, text=True, check=True
        )
        print(f"📸 Скриншот сохранен: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка скриншота:\n{e.stderr}", file=sys.stderr)



if 'mlflow_process' in locals() and mlflow_process.poll() is None:
    mlflow_process.terminate()
    mlflow_process.wait()

port = 5000
mlflow_tracking_uri = "file:./mlruns"
experiment_name = "Dumplings"

mlflow_process = subprocess.Popen(
    ["mlflow", "ui", "--port", str(port), "--backend-store-uri", mlflow_tracking_uri],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
print(f"🚀 MLflow UI запущен на http://localhost:{port}")
print(f"PID: {mlflow_process.pid}")

# Настраиваем клиент
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(experiment_name)

# ==================================================

NUM_CLASSES = dm.num_classes
MAX_EPOCHS = 15
BATCH_SIZE = 64
LR = 5e-5


filename = f'VGGonCLEAN-{time.localtime()[2]}.{time.localtime()[1]}-{time.localtime()[3]}.{time.localtime()[4]}-'


checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINTS_KR,
    filename=filename+"{epoch:02d}-{val_acc:.4f}",
    save_top_k=1,
    monitor="val_acc",
    mode="max"
)

# model_partial_ft = MODEL(num_classes=NUM_CLASSES,model_name='convnext_base.fb_in22k_ft_in1k_384', learning_rate= LR)
model_partial_ft = MODEL(num_classes=NUM_CLASSES, learning_rate= LR)
dm = DataModule(data_path=TRAIN_PATH,test_path=VAL_PATH, batch_size=BATCH_SIZE, num_workers=10)

dm.setup()

# early_stop_callback = EarlyStopping(monitor='val_acc', patience=7, verbose=True, mode='max')
trainer_partial_ft = pl.Trainer(
    max_epochs=MAX_EPOCHS, 
    accelerator="auto", 
    callbacks=[checkpoint_callback], # early_stop_callback # TQDMProgressBar(refresh_rate=10), 
    logger=[MLFlowLogger(experiment_name="Dumplings", tracking_uri="file:./mlruns", run_name=filename), CSVLogger(save_dir='dumplings', name=filename)])

trainer_partial_ft.fit(model_partial_ft, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.test_dataloader())

# ==================================================

trainer_partial_ft = pl.Trainer(
    max_epochs=8, 
    accelerator="auto", 
    callbacks=[checkpoint_callback], # early_stop_callback # TQDMProgressBar(refresh_rate=10), 
    logger=[MLFlowLogger(experiment_name="Dumplings", tracking_uri="file:./mlruns", run_name=filename), CSVLogger(save_dir='dumplings', name=filename)])

trainer_partial_ft.fit(model_partial_ft, train_dataloaders=dm.test_dataloader(), val_dataloaders=dm.test_dataloader())

# ==================================================

from tqdm.auto import tqdm


cur = checkpoint_callback.best_model_path.split('\\')[-1]

loaded_m = MODEL.load_from_checkpoint(CHECKPOINTS_KR/cur)

test_transform = T.Compose([
    T.Resize((224, 224)),
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

files = list(map(lambda x: x[0].split('\\')[-1], test_dataset.imgs))


ans = list(map(lambda x:dict(list(zip(range(dm.num_classes), dm.class_names))).get(x,None) , F.softmax(final_predictions, dim=1).argmax(dim=1).tolist()))

pd.DataFrame(pd.DataFrame([files, ans],index=['filename', 'class']).T).to_csv(f'submissions/{cur}.csv',index=False)

# ==================================================

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
    
run_full_evaluation(loaded_m, dm.test_dataloader(), '.', model_partial_ft.history, dm.class_names,{'device':device}, True)

# ==================================================

