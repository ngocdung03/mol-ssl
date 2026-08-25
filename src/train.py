"""Supervised training loop -- config-driven, seeded, ledger-logged.

Design constraints that are not negotiable (CLAUDE.md hard rules):
  * The split comes from a committed manifest and is always a scaffold split.
  * Model selection uses the validation split; the test split is touched exactly once, at the end.
  * The reported epoch is the early-stopping epoch, never the best test epoch.
  * Every run writes a metrics JSON and one ledger row, tagged with a run keyword.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.datamodule import build_loaders
from src.eval import (
    expected_calibration_error,
    masked_auroc,
    masked_rmse,
    selective_accuracy,
)
from src.models.gnn import PropertyPredictor


def load_pretrained_encoder(path: str, mcfg: dict) -> dict:
    """Encoder weights from a pretraining checkpoint, with an architecture-match check.

    Silently loading a mismatched encoder is the kind of bug that produces a plausible but
    meaningless SSL comparison, so a mismatch raises instead of being coerced.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for field, default in (("hidden", 300), ("layers", 5)):
        want = int(mcfg.get(field, default))
        got = int(ckpt.get(field, want))
        if want != got:
            raise ValueError(
                f"pretrained encoder {field}={got} but config asks for {field}={want}; "
                f"refusing to load a mismatched encoder ({path})"
            )
    return ckpt["encoder_state"]



def masked_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCE over labeled entries only. NaN targets are missing labels, not negatives."""
    mask = ~torch.isnan(y)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask])


def masked_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(y)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], y[mask])


@torch.no_grad()
def predict(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, outs = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        ys.append(batch.y.detach().cpu().numpy())
        outs.append(out.detach().float().cpu().numpy())
    if not ys:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.concatenate(ys), np.concatenate(outs)


def evaluate(model, loader, task: str, device: str) -> dict:
    y, out = predict(model, loader, device)
    if y.size == 0:
        return {}
    if task == "classification":
        prob = 1.0 / (1.0 + np.exp(-out))
        auroc, per_task, skipped = masked_auroc(y, prob)
        return {
            "auroc": auroc,
            "auroc_per_task": per_task,
            "n_tasks_skipped": skipped,
            "ece": expected_calibration_error(y, prob),
            "selective_accuracy": selective_accuracy(y, prob),
        }
    return {"rmse": masked_rmse(y, out)}


def train_one_seed(cfg: dict, seed: int, device: str | None = None, verbose: bool = True) -> dict:
    """Train a single seed end to end. Returns the metrics dict for that seed."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    task = cfg["task"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, info = build_loaders(cfg, seed=seed)

    mcfg = cfg.get("model", {})
    tcfg = cfg.get("train", {})
    model = PropertyPredictor(
        n_tasks=int(cfg.get("n_tasks", 1)),
        hidden=int(mcfg.get("hidden", 300)),
        layers=int(mcfg.get("layers", 5)),
        dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)

    # SSL: load pretrained encoder weights, discard the pretraining head. The prediction head stays
    # randomly initialized -- transferring it would leak the pretext task into the downstream one.
    ckpt_path = cfg.get("ssl", {}).get("checkpoint")
    if ckpt_path:
        model.encoder.load_state_dict(load_pretrained_encoder(ckpt_path, mcfg))
        info_pretrained = ckpt_path
    else:
        info_pretrained = None

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    # T4 is sm_75: fp16 or fp32 only, never bf16.
    use_amp = str(tcfg.get("amp", "fp16")) == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = masked_bce if task == "classification" else masked_mse
    better = (lambda a, b: a > b) if task == "classification" else (lambda a, b: a < b)
    key = "auroc" if task == "classification" else "rmse"

    epochs = int(tcfg.get("epochs", 100))
    patience = int(tcfg.get("patience", 20))
    best_val = -float("inf") if task == "classification" else float("inf")
    best_state, best_epoch, since_improve = None, -1, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(batch), batch.y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        val = evaluate(model, val_loader, task, device)
        score = val.get(key, float("nan"))
        if not np.isnan(score) and better(score, best_val):
            best_val, best_epoch, since_improve = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"  seed {seed} epoch {epoch:3d} val_{key} {score:.4f} (best {best_val:.4f} @ {best_epoch})")
        if since_improve >= patience:
            if verbose:
                print(f"  seed {seed} early stop at epoch {epoch} (patience {patience})")
            break

    # Restore the validation-selected checkpoint, then touch the test split exactly once.
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, task, device)

    return {
        "seed": seed,
        "selected_epoch": best_epoch,
        f"val_{key}": best_val,
        "test": test,
        "data": info,
        "pretrained_from": info_pretrained,
    }
