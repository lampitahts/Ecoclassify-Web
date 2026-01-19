"""
Robust training script for fine-tuning ResNet on the project dataset.

Features:
- Uses ImageFolder dataset (folder-per-class). Accepts arbitrary per-category folders.
- Stronger augmentation for robustness.
- Automatic class-weighting for imbalanced datasets.
- Mixed-precision training when CUDA available.
- Checkpointing (best validation accuracy) saved to backend/model.pth by default.
- Saves `class_to_idx` and `model_type` in the checkpoint for inference.

Usage (PowerShell example):
  python train.py --data_dir "..\dataset" --model efficientnet_b0 --epochs 15 --batch_size 16

After training, move the produced model.pth to backend/ (or run the script from backend so it writes there).
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
from collections import Counter

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
try:
    import timm
except Exception:
    timm = None

# Albumentations is optional; if missing, fall back to torchvision transforms.
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except Exception:
    A = None
    ToTensorV2 = None
    _HAS_ALB = False
from torch.utils.data import DataLoader, random_split


def get_transforms(img_size=224):
    if _HAS_ALB:
        # use albumentations for stronger augmentation
        train_aug = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
        val_aug = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
        return train_aug, val_aug
    else:
        # fallback to torchvision transforms (works with CPU-only installs)
        train_t = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
        val_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
        return train_t, val_t


# ...existing code...


def compute_class_weights(dataset):
    # dataset.samples -> list of (path, class_idx)
    from collections import Counter
    # support custom dataset that stores .targets (list of class_idx)
    if hasattr(dataset, 'targets') and isinstance(getattr(dataset, 'targets'), (list, tuple)):
        counts = Counter(dataset.targets)
    else:
        # torchvision.datasets.ImageFolder stores samples as (path, class_idx)
        try:
            counts = Counter([s[1] for s in dataset.samples])
        except Exception:
            # fallback: try to infer from dataset.classes
            num_classes = len(getattr(dataset, 'classes', []))
            counts = Counter({i: 1 for i in range(num_classes)})
    num_samples = sum(counts.values())
    # weight = N / (C * count_i)
    num_classes = len(counts)
    weights = {cls: num_samples / (num_classes * cnt) for cls, cnt in counts.items()}
    weight_list = [weights[i] for i in range(num_classes)]
    return torch.tensor(weight_list, dtype=torch.float)


def focal_loss(inputs, targets, gamma=2.0, alpha=None):
    """Simple focal loss for multi-class classification.
    inputs: logits (batch, C)
    targets: ground-truth (batch,)
    """
    ce = nn.CrossEntropyLoss(reduction='none')
    logpt = -ce(inputs, targets)
    pt = torch.exp(logpt)
    if alpha is not None:
        at = alpha[targets].to(inputs.device)
        loss = -at * ((1 - pt) ** gamma) * logpt
    else:
        loss = -((1 - pt) ** gamma) * logpt
    return loss.mean()


class AlbDataset(torch.utils.data.Dataset):
    """ImageFolder-like dataset that supports nested class folders.

    It will discover leaf directories containing images and treat each leaf as a class.
    Implemented at module level so it is picklable by DataLoader on Windows.
    """
    def __init__(self, folder, transform=None):
        self.samples = []  # list of (path, class_name)
        self.targets = []  # list of class_idx
        self.transform = transform
        # discover classes by walking folder and taking leaf dirs that contain image files
        classes = []
        folder = os.path.abspath(folder)
        for root, dirs, files in os.walk(folder):
            # check if this dir contains at least one image file
            imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(imgs) == 0:
                continue
            # compute a class label relative to the base folder
            rel = os.path.relpath(root, folder)
            # normalize label: replace os.sep with '_' to have flat class names
            class_name = rel.replace(os.sep, '_') if rel != '.' else os.path.basename(folder)
            classes.append(class_name)
            for f in imgs:
                self.samples.append(os.path.join(root, f))

        # deduplicate and sort classes to ensure stable ordering
        classes = sorted(set(classes))
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # now build targets by mapping each sample path to its class idx
        self.targets = []
        for p in self.samples:
            rel = os.path.relpath(os.path.dirname(p), folder)
            class_name = rel.replace(os.sep, '_') if rel != '.' else os.path.basename(folder)
            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(self.samples[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[idx]
        if self.transform:
            # albumentations returns dict with 'image' key when used
            try:
                data = self.transform(image=img)
                img = data['image']
            except Exception:
                # torchvision transforms accept PIL/numpy; convert to PIL if needed
                from PIL import Image
                img = Image.fromarray(img)
                img = self.transform(img)
        return img, target


def train(data_dir, model_name='efficientnet_b0', epochs=10, batch_size=16, lr=1e-4, out_path='backend/model.pth', seed=42, use_sampler=False, use_focal=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    torch.manual_seed(seed)

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Data dir not found: {data_dir}')

    # choose architecture and default input size
    arch = model_name
    if arch.startswith('efficientnet_b0'):
        img_size = 224
    elif arch.startswith('efficientnet_b3'):
        img_size = 300
    else:
        img_size = 224

    train_aug, val_aug = get_transforms(img_size=img_size)

    # use the module-level AlbDataset which handles both albumentations and torchvision transforms
    full_dataset = AlbDataset(str(data_dir), transform=train_aug)
    class_to_idx = full_dataset.class_to_idx
    # normalize class names to lowercase when storing mapping to avoid case issues
    class_to_idx_norm = {k.lower(): v for k, v in class_to_idx.items()}
    print('Found classes:', class_to_idx)

    # split into train/val (stratified-ish by using random_split with fixed seed)
    n = len(full_dataset)
    val_size = max( int(0.15 * n), 1)
    train_size = n - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    # replace val subset with dataset that uses val_aug transforms
    # Build a separate AlbDataset with val transforms
    val_dataset_full = AlbDataset(str(data_dir), transform=val_aug)
    val_ds = torch.utils.data.Subset(val_dataset_full, val_ds.indices)

    # On Windows, DataLoader multiprocessing often fails to pickle local objects
    # or transforms — use single-process data loading there.
    if os.name == 'nt':
        num_workers = 0
    else:
        num_workers = min(4, os.cpu_count() or 1)
    # optionally use WeightedRandomSampler to balance training
    if use_sampler:
        from torch.utils.data import WeightedRandomSampler
        # compute sample weights based on the training subset targets (train_ds.indices)
        train_indices = list(train_ds.indices) if hasattr(train_ds, 'indices') else list(range(len(train_ds)))
        train_targets = [full_dataset.targets[i] for i in train_indices]
        counts = Counter(train_targets)
        # build class weights using counts over the training subset
        num_samples = len(train_targets)
        class_weights = {cls: num_samples / (len(counts) * cnt) for cls, cnt in counts.items()}
        samples_weight = [class_weights[t] for t in train_targets]
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # build model
    # build model (EfficientNet or other timm architectures)
    num_classes = len(class_to_idx)
    if timm is None:
        raise RuntimeError('timm is required for EfficientNet training. Please pip install timm')

    if arch in ('efficientnet_b0', 'efficientnet_b3'):
        model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    else:
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)


    # criterion with class weights (used only for CrossEntropyLoss)
    ce_class_weights = compute_class_weights(full_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    if use_focal:
                        loss = focal_loss(outputs, labels, gamma=2.0, alpha=None)
                    else:
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                if use_focal:
                    loss = focal_loss(outputs, labels, gamma=2.0, alpha=None)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
        val_loss = running_loss / total
        val_acc = correct / total

        scheduler.step(val_acc)

        print(f'Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} time: {time.time()-t0:.1f}s')

        # checkpoint best
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            ckpt = {
                'model_state_dict': model.state_dict(),
                # store normalized mapping (keys lowercased)
                'class_to_idx': class_to_idx_norm,
                'model_type': model_name
            }
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            torch.save(ckpt, out_path)
            print('Saved best model to', out_path)

    # Always save latest checkpoint as well (helps ensure a checkpoint exists after short runs)
    try:
        last_ckpt = {
            'model_state_dict': model.state_dict(),
            'class_to_idx': class_to_idx_norm,
            'model_type': model_name,
            'last_epoch': epoch
        }
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        last_path = out_path
        torch.save(last_ckpt, last_path)
        print('Saved final model checkpoint to', last_path)
    except Exception as e:
        print('Failed to save final checkpoint:', e)

    print('Training finished. Best val acc:', best_acc, 'at epoch', best_epoch)
    # Always save final checkpoint (even if not the best) so backend can load a model file for testing.
    try:
        final_ckpt = {
            'model_state_dict': model.state_dict(),
            'class_to_idx': class_to_idx,
            'model_type': model_name,
            'best_val_acc': best_acc,
            'best_epoch': best_epoch
        }
        final_path = out_path + '.last'
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        torch.save(final_ckpt, final_path)
        print('Saved final checkpoint to', final_path)
    except Exception as e:
        print('Failed to save final checkpoint:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', choices=[ 'efficientnet_b0', 'efficientnet_b3'], default='efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', type=str, default='backend/model.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_sampler', action='store_true', help='Use WeightedRandomSampler to balance training')
    parser.add_argument('--focal_loss', action='store_true', help='Use focal loss instead of CrossEntropyLoss')
    args = parser.parse_args()
    train(args.data_dir, model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_path=args.out, seed=args.seed, use_sampler=args.use_sampler, use_focal=args.focal_loss)
