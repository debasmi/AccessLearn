"""
Indian Sign Language (ISL) Recognition - Training Pipeline
============================================================
Uses MediaPipe Hand Landmarks + MLP Classifier (no GPU needed)
"""
import os
import sys
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LANDMARK_FEATURES = 21 * 3       # 21 landmarks × (x, y, z)
NORMALIZED_FEATURES = 21 * 2     # 21 landmarks × (x_norm, y_norm) after wrist-relative


def extract_landmarks(image_path: str) -> np.ndarray | None:
   
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0].landmark
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])

    xs -= xs[0];  ys -= ys[0]
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
    xs /= span;   ys /= span

    return np.concatenate([xs, ys]).astype(np.float32)


def build_dataset(data_dir: str):
    
    X, y = [], []
    classes = sorted(os.listdir(data_dir))

    print(f"\n📂  Found {len(classes)} class folders: {classes}\n")

    failed = 0
    for cls in tqdm(classes, desc="Extracting landmarks"):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTENSIONS:
                continue
            fpath = os.path.join(cls_dir, fname)
            feats = extract_landmarks(fpath)
            if feats is not None:
                X.append(feats)
                y.append(cls)
            else:
                failed += 1

    print(f"\n✅  Extracted {len(X)} samples  |  ❌ Failed: {failed} (no hand detected)")
    return np.array(X, dtype=np.float32), np.array(y)


class ISLNet(nn.Module):
    """Lightweight MLP for landmark-based sign classification."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * len(xb)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += len(xb)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_true


def plot_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_acc"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"📊  Training curves saved → {save_path}")


def plot_confusion(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"📊  Confusion matrix saved → {save_path}")



def main():
    parser = argparse.ArgumentParser(description="ISL Sign Language Trainer")
    parser.add_argument("--data_dir",   required=True, help="Path to Indian/ dataset folder")
    parser.add_argument("--output_dir", default="isl_model", help="Where to save model & artifacts")
    parser.add_argument("--epochs",     type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--test_size",  type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️   Device: {device}")

    # ──features ─────────────────────────────────────────────────────
    X, y_raw = build_dataset(args.data_dir)
    if len(X) == 0:
        print("❌  No samples extracted. Check your data_dir path.")
        sys.exit(1)

    # ── Encoding ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"\n🔤  Classes ({num_classes}): {list(le.classes_)}")

    # ── Train / val split ─────────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"📊  Train: {len(X_tr)} | Val: {len(X_val)}")

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # ── model ───────────────────────────────────────────────────────────
    model = ISLNet(input_dim=NORMALIZED_FEATURES, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "isl_best_model.pt")

    print(f"\n🚀  Training for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={tr_loss:.4f}/{vl_loss:.4f}  "
                  f"acc={tr_acc:.3f}/{vl_acc:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    print(f"\n🏆  Best Val Accuracy: {best_val_acc:.4f}")

  
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, _, preds, trues = evaluate(model, val_loader, criterion, device)

    print("\n📋  Classification Report:\n")
    print(classification_report(trues, preds, target_names=le.classes_))

    cm = confusion_matrix(trues, preds)
    plot_curves(history, os.path.join(args.output_dir, "training_curves.png"))
    plot_confusion(cm, le.classes_, os.path.join(args.output_dir, "confusion_matrix.png"))

   
    with open(os.path.join(args.output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    config = {
        "num_classes": num_classes,
        "classes": list(le.classes_),
        "input_dim": NORMALIZED_FEATURES,
        "best_val_acc": best_val_acc,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅  All files saved in: {args.output_dir}/")
    print(f"    ├── isl_best_model.pt    (model weights)")
    print(f"    ├── label_encoder.pkl    (class names)")
    print(f"    ├── config.json          (metadata)")
    print(f"    ├── training_curves.png")
    print(f"    └── confusion_matrix.png")
    print(f"\n➡️   Run inference with: python isl_predict.py --model_dir {args.output_dir} --image path/to/sign.jpg\n")


if __name__ == "__main__":
    main()
