"""CNN model for attendance cell classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttendanceCNN(nn.Module):
    """Small CNN for classifying attendance cells.

    Input: 1x32x32 grayscale cell image
    Output: 3 classes (A, BLANK, UNKNOWN)
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        # After 3 pool layers: 32/2/2/2 = 4, so 64*4*4 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32→16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16→8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8→4
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CellDataset(torch.utils.data.Dataset):
    """Dataset for training the cell classifier.

    Expects a directory structure:
        data_dir/
            A/
                cell_001.png
                ...
            BLANK/
                cell_001.png
                ...
            UNKNOWN/
                cell_001.png
                ...
    """

    def __init__(self, data_dir: str, transform=None):
        import os
        from pathlib import Path

        self.samples = []
        self.transform = transform
        self.label_map = {"A": 0, "BLANK": 1, "UNKNOWN": 2}

        root = Path(data_dir)
        for label_name, label_idx in self.label_map.items():
            label_dir = root / label_name
            if not label_dir.exists():
                continue
            for img_path in label_dir.glob("*.png"):
                self.samples.append((str(img_path), label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import cv2
        import numpy as np

        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((32, 32), dtype=np.uint8)
        img = cv2.resize(img, (32, 32))
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


def train_model(
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    save_path: str = "models/cell_classifier.pth",
) -> AttendanceCNN:
    """Train the CNN classifier on labeled cell images."""
    from torch.utils.data import DataLoader, random_split
    from pathlib import Path

    dataset = CellDataset(data_dir)
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {data_dir}")

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = AttendanceCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(total, 1)
        print(
            f"Epoch {epoch+1}/{epochs} — Loss: {total_loss:.4f}, Val Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best val accuracy: {best_val_acc:.3f}")
    return model
