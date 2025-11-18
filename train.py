import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

from config import cfg
from data import SodaCanDataset, get_transforms
from models import ResNetCanClassifier
from training import CanRotationTrainer


def main():
    parser = argparse.ArgumentParser(description='Train soda can rotation model')
    parser.add_argument('--epochs', type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=cfg.LEARNING_RATE)
    args = parser.parse_args()

    print("Initializing Soda Can Rotation Training...")
    print(f"Device: {cfg.DEVICE}")
    print(f"Data directory: {cfg.SNAPS_DIR}")

    if not cfg.SNAPS_DIR.exists():
        raise FileNotFoundError(f"Data directory {cfg.SNAPS_DIR} not found!")
    if not cfg.LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels file {cfg.LABELS_FILE} not found!")

    full_dataset = SodaCanDataset(
        data_dir=cfg.SNAPS_DIR,
        labels_file=cfg.LABELS_FILE,
        phase='train',
        transform=get_transforms('train')
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = get_transforms('val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    model = ResNetCanClassifier(
        num_classes=cfg.NUM_CLASSES,
        backbone=cfg.MODEL_NAME
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = CanRotationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.DEVICE
    )

    trainer.train(args.epochs)

    print("Training completed!")
    print(f"Best validation accuracy: {trainer.best_accuracy:.2f}%")


if __name__ == '__main__':
    main()