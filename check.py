# check.py
import torch
from torch.utils.data import Dataset, DataLoader

from models import build_model
from experiments import train_model, evaluate_model, device   # <- import device


# ---- Synthetic Dataset ----
class SyntheticVadDataset(Dataset):
    def __init__(self, n_samples=8, T=20, F=40):
        self.n_samples = n_samples
        self.T = T
        self.F = F

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.randn(self.T, self.F)          # [T, F]
        y = torch.randint(0, 2, (self.T,))       # [T]
        length = self.T
        return x, y, length


def simple_collate(batch):
    xs, ys, lengths = zip(*batch)   # each x: [T,F], y: [T]
    xb = torch.stack(xs, dim=0)     # [B, T, F]
    yb = torch.stack(ys, dim=0)     # [B, T]
    lengths = torch.tensor(lengths) # [B]
    return xb, yb, lengths


def make_loader(n_samples=8, batch_size=4, T=20, F=40):
    ds = SyntheticVadDataset(n_samples=n_samples, T=T, F=F)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=simple_collate)
    return dl


def main():
    n_features = 40
    hidden_size = 32
    num_layers = 1

    train_dl = make_loader(n_samples=16, batch_size=4, T=20, F=n_features)
    val_dl   = make_loader(n_samples=8,  batch_size=4, T=20, F=n_features)
    test_dl  = make_loader(n_samples=8,  batch_size=4, T=20, F=n_features)

    for model_type in ["lstm", "bilstm", "cnnlstm"]:
        print(f"\n=== Testing model_type = {model_type} ===")
        model = build_model(
            model_type,
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        # 🔴 This is the missing piece:
        model = model.to(device)

        model, hist = train_model(
            model,
            train_dl,
            val_dl,
            epochs=1,
            lr=1e-3,
            patience=1,
        )

        acc, prec, rec = evaluate_model(model, test_dl)
        print(f"Test OK → Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")


if __name__ == "__main__":
    main()
