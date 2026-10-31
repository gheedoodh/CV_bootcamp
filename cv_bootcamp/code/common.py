import argparse
import copy
import importlib.util
import json
import os
import platform
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def add_common_args(parser):
    parser.add_argument("--data_dir", default=".", help="folder with the provided files, including model.py")
    parser.add_argument("--out_dir", default=".", help="folder that receives predictions/, logs/ and results")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--scheduler_tmax", type=int, default=None,
                        help="cosine T_max; defaults to --epochs")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="epochs of plain CE before the robust loss kicks in")
    return parser


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dirs(out_dir):
    for sub in ("predictions", "logs", "checkpoints"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def env_info():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def load_data(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    assert (df["index"].values == np.arange(len(df))).all()

    all_images = np.load(os.path.join(data_dir, "train_images.npy"))
    train_idx = np.loadtxt(os.path.join(data_dir, "train_idx.txt"), dtype=int)
    val_idx = np.loadtxt(os.path.join(data_dir, "val_idx.txt"), dtype=int)
    test_images = np.load(os.path.join(data_dir, "test_images.npy"))

    assert all_images.shape == (50000, 32, 32, 3), all_images.shape
    assert test_images.shape == (10000, 32, 32, 3), test_images.shape
    assert all_images.dtype == np.uint8 and test_images.dtype == np.uint8
    assert len(train_idx) == 45000 and len(val_idx) == 5000
    assert len(set(train_idx) & set(val_idx)) == 0

    train_images = all_images[train_idx]
    val_images = all_images[val_idx]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    mean = [float(train_images[..., c].astype(np.float64).mean() / 255) for c in range(3)]
    std = [float(train_images[..., c].astype(np.float64).std() / 255) for c in range(3)]

    return SimpleNamespace(
        train_idx=train_idx,
        val_idx=val_idx,
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        train_df=train_df,
        val_df=val_df,
        train_noisy=train_df["noisy_label"].to_numpy(),
        train_synth=train_df["synthetic_label"].to_numpy(),
        train_clean=train_df["clean_label"].to_numpy(),
        val_noisy=val_df["noisy_label"].to_numpy(),
        val_synth=val_df["synthetic_label"].to_numpy(),
        val_clean=val_df["clean_label"].to_numpy(),
        mean=mean,
        std=std,
    )


def make_transforms(mean, std):
    train_tf = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


class LabeledImages(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.transform(self.images[index]), int(self.labels[index]), index


class UnlabeledImages(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.transform(self.images[index])


def make_eval_loader(images, eval_tf, batch_size=256):
    return DataLoader(UnlabeledImages(images, eval_tf), batch_size=batch_size, shuffle=False)


def load_resnet18(data_dir):
    """Load the provided ResNet18 from any of the layouts the bootcamp allows.

    Expected project layout:

        <project_root>/
        ├── model.py       <- provided model
        ├── code/          <- these scripts
        └── data/          <- labels.csv, *.npy, *.txt

    `--data_dir` points at data/. The search below also covers the flattened
    layout (model.py inside data/) and the case where the user placed
    model.py next to the scripts. First existing path wins.
    """
    abs_data = os.path.abspath(data_dir)
    abs_data_parent = os.path.dirname(abs_data)
    abs_script_parent = os.path.dirname(_SCRIPT_DIR)

    candidates = [
        os.path.join(abs_data, "model.py"),          # model.py inside data/
        os.path.join(abs_data_parent, "model.py"),   # <project_root>/model.py (recommended)
        os.path.join(_SCRIPT_DIR, "model.py"),       # model.py next to scripts
        os.path.join(abs_script_parent, "model.py"), # <project_root>/model.py via script parent
        os.path.abspath("model.py"),                 # CWD fallback
    ]

    seen, unique = set(), []
    for p in candidates:
        key = os.path.normcase(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    for path in unique:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("provided_model", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "ResNet18"):
                raise AttributeError(
                    f"{path} does not define ResNet18 — is this the provided model.py?"
                )
            return module.ResNet18

    raise FileNotFoundError(
        "model.py not found. Searched:\n  " + "\n  ".join(unique)
    )


def build_model(resnet_cls, device):
    return resnet_cls(num_classes=10).to(device)


class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_y_q = torch.exp(self.q * target_log_probs)
        return ((1.0 - p_y_q) / self.q).mean()



class WarmupThenLoss(nn.Module):
    def __init__(self, warmup_epochs, robust_loss):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.robust_loss = robust_loss
        self.warmup_loss = nn.CrossEntropyLoss()
        self.epoch = 0
        self.needs_index = getattr(robust_loss, "needs_index", False)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, logits, targets, index=None):
        if self.epoch < self.warmup_epochs:
            return self.warmup_loss(logits, targets)
        if self.needs_index:
            if index is None:
                return self.warmup_loss(logits, targets)
            return self.robust_loss(logits, targets, index)
        return self.robust_loss(logits, targets)


class ELRLoss(nn.Module):
    needs_index = True

    def __init__(self, num_examples, num_classes=10, lam=3.0, beta=0.7):
        super().__init__()
        self.register_buffer("target", torch.zeros(num_examples, num_classes))
        self.lam = lam
        self.beta = beta

    def forward(self, logits, labels, index):
        y_pred = F.softmax(logits, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            y_det = y_pred.detach()
            y_det = y_det / y_det.sum(dim=1, keepdim=True)
            self.target[index] = self.beta * self.target[index] + (1.0 - self.beta) * y_det
        ce = F.cross_entropy(logits, labels)
        reg = torch.log(1.0 - (self.target[index] * y_pred).sum(dim=1)).mean()
        return ce + self.lam * reg


def train_one_epoch(model, loader, optimizer, loss_fn, device, correct_row=None):
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    needs_index = getattr(loss_fn, "needs_index", False)
    for images, labels, index in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        index = index.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels, index) if needs_index else loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        with torch.no_grad():
            ok = logits.argmax(dim=1) == labels
            correct += ok.sum().item()
            if correct_row is not None:
                correct_row[index.cpu().numpy()] = ok.cpu().numpy()
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(model(images.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def logits_on(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(model(images.to(device)).float().cpu())
    return torch.cat(out)


def per_sample_ce(logits, labels):
    return F.cross_entropy(logits, torch.as_tensor(labels, dtype=torch.long), reduction="none").numpy()


def make_folds(labels, seed, n_splits=5):
    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splits.split(np.zeros(len(labels)), np.asarray(labels)))


def run_training(args, tag, resnet_cls, device, train_images, train_labels, val_images, val_labels,
                 test_images, train_tf, eval_tf, loss_factory, select_from_epoch=0, track_train=True):
    """Train one network under the shared recipe.

    Only `train_labels` is used for the loss and only `val_labels` for epoch selection.
    Seeding order: seed, model, loss/optimizer, loader generator, seed again before the loop.
    """
    if args.deterministic and args.num_workers > 0:
        raise SystemExit("--deterministic requires --num_workers 0")

    out_logs = os.path.join(args.out_dir, "logs")
    out_ckpt = os.path.join(args.out_dir, "checkpoints")

    train_dataset = LabeledImages(train_images, train_labels, train_tf)
    val_loader = make_eval_loader(val_images, eval_tf)
    test_loader = make_eval_loader(test_images, eval_tf)

    set_seed(args.seed, args.deterministic)
    model = build_model(resnet_cls, device)
    loss_fn = loss_factory(len(train_dataset)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    tmax = args.scheduler_tmax if getattr(args, "scheduler_tmax", None) is not None else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        generator=g, num_workers=args.num_workers)

    _, check_labels, check_index = next(iter(loader))
    assert (check_labels.numpy() == np.asarray(train_labels)[check_index.numpy()]).all()

    g.manual_seed(args.seed)
    set_seed(args.seed, args.deterministic)

    best_val = -1.0
    best_epoch = 0
    best_state = None
    history = []
    train_correct = np.zeros((args.epochs, len(train_dataset)), dtype=bool) if track_train else None
    val_preds = np.zeros((args.epochs, len(val_images)), dtype=np.int8)
    test_preds = np.zeros((args.epochs, len(test_images)), dtype=np.int8)

    for epoch in range(args.epochs):
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        row = train_correct[epoch] if track_train else None
        train_loss, train_acc = train_one_epoch(model, loader, optimizer, loss_fn, device, row)
        scheduler.step()

        epoch_val = predict(model, val_loader, device)
        val_acc = float((epoch_val == np.asarray(val_labels)).mean())
        val_preds[epoch] = epoch_val
        test_preds[epoch] = predict(model, test_loader, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_acc": val_acc, "lr": lr_now})
        print(f"[{tag} epoch {epoch}] loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} lr={lr_now:.4f}")

        if not np.isfinite(train_loss):
            print("non-finite loss, stopping")
            break

        if epoch >= select_from_epoch and val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        pd.DataFrame(history).to_csv(os.path.join(out_logs, f"{tag}_history.csv"), index=False)

    np.save(os.path.join(out_logs, f"{tag}_test_preds_all_epochs.npy"), test_preds)
    np.save(os.path.join(out_logs, f"{tag}_val_preds_all_epochs.npy"), val_preds)
    if track_train:
        np.save(os.path.join(out_logs, f"{tag}_train_correct_all_epochs.npy"), train_correct)

    final_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = final_state
        best_epoch = history[-1]["epoch"]
        best_val = history[-1]["val_acc"]

    model.load_state_dict(best_state)
    check_preds = predict(model, val_loader, device)
    assert abs(float((check_preds == np.asarray(val_labels)).mean()) - best_val) < 1e-9

    if args.save_checkpoints:
        torch.save(best_state, os.path.join(out_ckpt, f"{tag}_best.pt"))
        torch.save(final_state, os.path.join(out_ckpt, f"{tag}_last.pt"))

    return SimpleNamespace(
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_val=best_val,
        final_val=history[-1]["val_acc"],
        train_correct=train_correct,
        val_preds=val_preds,
        test_preds=test_preds,
    )


def write_predictions(preds, path):
    preds = np.asarray(preds).astype(int)
    assert preds.shape == (10000,) and preds.min() >= 0 and preds.max() <= 9
    pd.DataFrame({"id": np.arange(len(preds)), "pred": preds}).to_csv(path, index=False)


def posthoc_diagnostics(result, train_labels, train_clean, val_clean, path):
    """Clean labels are used here only, after training and selection are finished."""
    mislabeled = np.asarray(train_labels) != np.asarray(train_clean)
    rows = []
    for e in range(len(result.history)):
        rows.append({
            "epoch": e,
            "train_acc_correct": result.train_correct[e][~mislabeled].mean(),
            "train_acc_wrong": result.train_correct[e][mislabeled].mean(),
            "val_acc_clean": float((result.val_preds[e] == np.asarray(val_clean)).mean()),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return float(mislabeled.mean())



def make_dynamics_csv(history_df, diag_df, path):
    merged = diag_df.merge(history_df[["epoch", "val_acc"]], on="epoch", how="left")
    merged = merged.rename(columns={
        "train_acc_wrong": "train_acc_mislabeled",
        "val_acc": "val_acc_noisy",
    })
    cols = ["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]
    merged[cols].to_csv(path, index=False)


def update_results(path, section, values):
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results[section] = values
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)