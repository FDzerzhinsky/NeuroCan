import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from config.config import cfg

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path


class CanRotationTrainer:
    """Тренер для обучения модели определения угла поворота"""

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.best_accuracy = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Используем tqdm с минималистичным выводом
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:3d}',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                    leave=False)

        for batch in pbar:
            images = batch['image'].to(self.device)
            angles = batch['angle'].to(self.device)
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: убедимся, что углы в правильном диапазоне
            if torch.any(angles < 0) or torch.any(angles >= self.model.num_classes):
                invalid_indices = torch.where((angles < 0) | (angles >= self.model.num_classes))[0]
                print(f"ERROR: Invalid angles found in batch: {angles[invalid_indices]}")
                continue  # пропускаем проблемный батч
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, angles)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += angles.size(0)
            correct += (predicted == angles).sum().item()

            # Обновляем прогресс без лишней информации
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100 * correct / total:5.1f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val {epoch:3d}',
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                        leave=False)

            for batch in pbar:
                images = batch['image'].to(self.device)
                angles = batch['angle'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, angles)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += angles.size(0)
                correct += (predicted == angles).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'acc': f'{100 * correct / total:5.1f}%'
                })

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs on {self.device}")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            self.scheduler.step()

            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_checkpoint(epoch, is_best=True)

            # Чистый вывод без лишней информации
            print(f'Epoch {epoch:3d}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:5.1f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:5.1f}% | '
                  f'Best: {self.best_accuracy:5.1f}%')

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy
        }

        if is_best:
            best_filename = cfg.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_filename)
            print(f"Saved best model with accuracy {self.best_accuracy:.2f}%")