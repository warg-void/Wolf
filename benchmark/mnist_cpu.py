import os
import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

num_pixels = 784
num_classes = 10

def load_mnist_csv(path: str):
    xs, ys = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError("Empty MNIST file")

        for row_i, row in enumerate(reader):
            if not row:
                continue
            if len(row) != 1 + num_pixels:
                raise RuntimeError(f"Malformed MNIST row at {row_i} in {path}")

            label = int(row[0])
            pixels = np.asarray(row[1:], dtype=np.float32) / 255.0
            xs.append(pixels)
            ys.append(label)

    x = torch.from_numpy(np.stack(xs))          
    y = torch.tensor(ys, dtype=torch.long)      
    return x, y

class MnistCsvDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_pixels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)

def maybe_compile(model):
    # torch.compile is the closest thing to "-O3" for the *model graph* in PyTorch 2.x.
    if hasattr(torch, "compile"):
        # good default for CPU; you can try "max-autotune" too
        return torch.compile(model, mode="reduce-overhead")
    return model

def main():
    exe_dir = Path(__file__).resolve().parent
    data_dir = exe_dir / "data"
    train_path = data_dir / "mnist_train.csv"
    test_path  = data_dir / "mnist_test.csv"

    print(f"Executable dir: {exe_dir}")
    print(f"Loading MNIST train from: {train_path}")
    print(f"Loading MNIST test  from: {test_path}")

    x_train, y_train = load_mnist_csv(str(train_path))
    x_test,  y_test  = load_mnist_csv(str(test_path))

    n_train_samples = x_train.shape[0]
    n_test_samples  = x_test.shape[0]
    print(f"Loaded {n_train_samples} train samples, {n_test_samples} test samples")

    # ---- Force CPU only ----
    device = torch.device("cpu")


    model = MLP().to(device)
    model = maybe_compile(model)

    lr = 0.02
    epochs = 5
    batch_size = 5

    train_ds = MnistCsvDataset(x_train, y_train)
    test_ds  = MnistCsvDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    optim = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ---- Training ----
    for epoch in range(epochs):
        model.train()
        begin = time.time()
        epoch_loss, seen = 0.0, 0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            seen += bs


        avg_loss = epoch_loss / n_train_samples
        runtime_ms = int((time.time() - begin) * 1000)
        print(f"Epoch {epoch} finished. Avg loss = {avg_loss} runtime epoch = {runtime_ms}ms")

    # ---- Evaluation ----
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()

    acc = 100.0 * correct / n_test_samples
    print(f"Test accuracy: {correct}/{n_test_samples} ({acc:.2f}%)")

if __name__ == "__main__":
    main()
