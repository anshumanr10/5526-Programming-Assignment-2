# ============================================================
# experiments.py
# ============================================================
# Assignment: Implement the full training, validation, and evaluation
# loops for Voice Activity Detection (VAD) models.
#
# You must implement all parts except:
#   ✅ The alignment snippet (provided below)
#   ✅ The run_experiment() function (provided and complete)
#
# Use the alignment snippet exactly where specified to ensure
# model outputs and labels have matching time dimensions.
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from dataloader import create_vad_dataloaders
from models import build_model
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Device Setup
# ------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"))
print(f"✅ Using device: {device}")


# ============================================================
# Alignment Snippet (Use This As-Is)
# ============================================================
"""
Use the following snippet whenever you compare model outputs and labels
to compute the loss or predictions. It ensures both tensors have the
same time dimension length (T).

Place it right after obtaining `logits = model(xb)` and before computing the loss.

Example:
    logits = model(xb)
    # Align lengths (insert here)
    B, Tm, C = logits.shape
    Ty = yb.size(1)
    min_T = min(Tm, Ty)
    logits = logits[:, :min_T, :]
    yb = yb[:, :min_T]
"""

# ============================================================
# Training and Validation
# ============================================================
def train_model(model, train_dl, val_dl, epochs=20, lr=1e-3, patience=3):
    """
    Implement the full training loop with validation and early stopping.
    """
    # TODO: define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # TODO: define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # TODO: define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    # TODO: initialize early stopping variables and tracking lists
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    best_state = None


    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        # TODO: iterate over batches
        cumulative_loss = 0.0
        for xb, yb, lengths in train_dl:
            # TODO: move data to device
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()

            # TODO: forward pass (use the alignment snippet before computing loss)
            logits = model(xb)
            # Align lengths (insert here)
            B, Tm, C = logits.shape
            Ty = yb.size(1)
            min_T = min(Tm, Ty)
            logits = logits[:, :min_T, :]
            yb = yb[:, :min_T]

        # TODO: compute loss and backpropagate
            loss = criterion(logits.reshape(B * min_T, C), yb.reshape(B * min_T))
        # TODO: apply gradient clipping
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # TODO: optimizer step and track loss
            optimizer.step()
            cumulative_loss += loss.item()
        
        avg_train_loss = cumulative_loss / len(train_dl)
        train_losses.append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        # TODO: evaluate on validation set using same alignment snippet
        cumulative_val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb, lengths in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                # Align lengths (insert here)
                B, Tm, C = logits.shape
                Ty = yb.size(1)
                min_T = min(Tm, Ty)
                logits = logits[:, :min_T, :]
                yb = yb[:, :min_T]
            # TODO: collect predictions and compute validation loss
                loss = criterion(logits.reshape(B * min_T, C), yb.reshape(B * min_T))
                cumulative_val_loss += loss.item()

                # predictions [B, Tm] and mask out padding (-1)
                preds = logits.argmax(dim=-1)    # [B, Tm]
                mask = (yb != -1)
                all_preds.append(preds[mask].cpu())
                all_labels.append(yb[mask].cpu())

        avg_val_loss = cumulative_val_loss / len(val_dl)
        val_losses.append(avg_val_loss)

        # TODO: compute accuracy, precision, recall
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()

        val_acc = float((y_true == y_pred).mean())
        val_prec = precision_score(y_true, y_pred, zero_division=0)
        val_rec = recall_score(y_true, y_pred, zero_division=0)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} "
            f"(Prec: {val_prec:.4f}, Rec: {val_rec:.4f})"
        )

        # TODO: scheduler step
        scheduler.step(avg_val_loss)
        # TODO: implement early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("EARLY STOPPING TRIGGERED")
                break

    # TODO: load best model before returning
    model.load_state_dict(best_state)
    # TODO: return model and training history (train_loss, val_loss)
    return model, {"train_loss": train_losses, "val_loss": val_losses}


# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model, test_dl):
    """
    Evaluate the model on the test set.

    Steps:
      - Set model to evaluation mode
      - Iterate through test batches
      - Use alignment snippet before comparing predictions and labels
      - Compute accuracy, precision, and recall
    """
    # TODO: implement evaluation loop
    model.eval()
    all_preds = []
    all_labels = []

    # TODO: collect predictions and labels
    with torch.no_grad():
        for xb, yb, lengths in test_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            # Align lengths (insert here)
            B, Tm, C = logits.shape
            Ty = yb.size(1)
            min_T = min(Tm, Ty)
            logits = logits[:, :min_T, :]
            yb_trunc = yb[:, :min_T]

            preds = logits.argmax(dim=-1)  # [B, Tm]
            mask = (yb_trunc != -1)
            all_preds.append(preds[mask].cpu())
            all_labels.append(yb_trunc[mask].cpu())

    # TODO: compute metrics
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = float((y_true == y_pred).mean())
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return acc, prec, rec


# ============================================================
# Experiment Runner (Provided)
# ============================================================
def run_experiment(model_name, n_features=40, hidden_size=128, layers=2, seq_len=25, snr_db=-10):
    print(f"\n🚀 Running {model_name.upper()} model (seq_len={seq_len})...")

    train_dl, val_dl, test_dl = create_vad_dataloaders(
        data_root="SpeechCommands/speech",
        noise_train="SpeechCommands/noise",
        noise_val="SpeechCommands/noise",
        noise_test="SpeechCommands/noise",
        batch_size=64,
        num_workers=8,
        seq_len=seq_len,
    )

    model = build_model(model_name, n_features=n_features, hidden_size=hidden_size, num_layers=layers)
    model = model.to(device, non_blocking=True)

    model, hist = train_model(model, train_dl, val_dl, lr=1e-3)

    plt.figure(figsize=(6, 4))
    plt.plot(hist["train_loss"], label="Train")
    plt.plot(hist["val_loss"], label="Val")
    plt.title(f"{model_name.upper()} Learning Curve (seq={seq_len})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    acc, prec, rec = evaluate_model(model, test_dl)
    print(f"✅ Test → Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}")
    return acc, prec, rec
