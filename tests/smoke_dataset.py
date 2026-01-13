from data.transforms import get_transforms
from data.dataset import SodaCanDataset
from config.config import cfg
from torch.utils.data import random_split, DataLoader

print('CFG SNAPS_DIR=', cfg.SNAPS_DIR)

# Попробуйте использовать папку snaps_extended, если есть
import os
base_path = os.path.join(os.getcwd(), 'data', 'snaps_extended')
if not os.path.exists(base_path):
    # fallback to cfg.SNAPS_DIR
    base_path = str(cfg.SNAPS_DIR)

print('Using dataset path:', base_path)

transform = get_transforms('train')
ds = SodaCanDataset(base_path, labels_file=None, phase='train', transform=transform)
print('Loaded samples:', len(ds.samples))

train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = random_split(ds, [train_size, val_size])
print('Train subset size (subset.indices length):', len(getattr(train_ds, 'indices', [])))

# generate augmented entries for train indices
train_ds.dataset.generate_augmented_entries_for_indices(getattr(train_ds, 'indices', []))
print('Augmented entries generated:', len(train_ds.dataset.augmented_entries))
print('Total dataset length now:', len(train_ds.dataset))

# create DataLoader with num_workers=0 to avoid worker issues
dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(dl))
print('Batch keys:', list(batch.keys()))
for k, v in batch.items():
    try:
        print(k, type(v), hasattr(v, 'shape') and getattr(v, 'shape', None))
    except Exception:
        print(k, type(v))

print('SMOKE TEST OK')
