"""train.py — Hoàn thiện pipeline huấn luyện, đánh giá và thí nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import contextlib
import copy
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Giá trị khởi điểm, tinh chỉnh bằng val
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    was_training = model.training
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    if was_training:
        model.train()
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    was_training = model.training
    model.eval()
    n = len(X)
    total_loss = 0.0
    all_preds = []

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        if loss_name == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += loss.item()
        elif loss_name == "mse":
            yb_one_hot = F.one_hot(yb, num_classes=logits.size(-1)).to(dtype=logits.dtype)
            loss = F.mse_loss(logits, yb_one_hot, reduction="mean") * len(xb)
            total_loss += loss.item()
        else:
            raise ValueError(f"Khong ho tro loss: '{loss_name}'")

        all_preds.append(logits.argmax(dim=1))

    preds = torch.cat(all_preds, dim=0)
    avg_loss = float(total_loss / n)
    acc = float((preds == y).float().mean().item())

    # Ma tran nham lan 7x7
    y_true_np = y.cpu().numpy()
    y_pred_np = preds.cpu().numpy()
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_true_np, y_pred_np), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    if was_training:
        model.train()

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.size(-1)).to(dtype=logits.dtype)
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Khong ho tro loss: '{loss_name}'. Chon 'ce' hoac 'mse'.")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    """
    seed = cfg.get("seed", 1)
    set_seed(seed)

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init_method = cfg.get("init", "he")
    device = data["X_tr"].device

    model = MLP(hidden=hidden, dropout=dropout, init=init_method, in_features=54, num_classes=7).to(device)
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden]

    optimizer_name = cfg.get("optimizer", "sgd_momentum")
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr=lr,
                                weight_decay=weight_decay, momentum=momentum)

    precision = cfg.get("precision", "fp32")
    if precision == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    elif precision == "bf16" and device.type == "cuda":
        scaler = None
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_ctx = contextlib.nullcontext()

    step0_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_eval["loss"]

    # Tap con co dinh 50k mau de theo doi train loss o eval mode
    n_train_eval = min(50_000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:n_train_eval]
    y_tr_eval = data["y_tr"][:n_train_eval]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": []
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    diverged = False

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm", None)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    print(f"--- Bat dau train {cfg.get('exp_id')} ({epochs} epochs, opt={optimizer_name}, lr={lr}) ---")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        grad_norms_epoch = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size,
                                      generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Canh bao: Loss bi NaN/Inf tai epoch {epoch}!")
                diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            grad_norms_epoch.append(gn)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - t0

        if diverged:
            break

        train_eval = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
        avg_grad_norm = float(np.mean(grad_norms_epoch)) if grad_norms_epoch else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(avg_grad_norm)
        history["epoch_time_s"].append(epoch_time)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_val_acc = val_eval["acc"]
            best_val_macro_f1 = val_eval["macro_f1"]
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | "
                  f"Train Loss: {train_eval['loss']:.4f} | "
                  f"Val Loss: {val_eval['loss']:.4f} | "
                  f"Val Acc: {val_eval['acc']:.4f} | "
                  f"Val F1: {val_eval['macro_f1']:.4f} | "
                  f"Grad: {avg_grad_norm:.2f} | "
                  f"Time: {epoch_time:.2f}s")

    peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    time_per_epoch = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(time_per_epoch),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "row_id": np.asarray(row_id, dtype=np.int64),
        "pred": np.asarray(preds, dtype=np.int64)
    })
    df.to_csv(p, index=False)
    print(f"-> Da ghi {len(df)} dong du doan vao: {p}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    model = MLP(hidden=hidden, dropout=0.0, in_features=54, num_classes=7).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()

    print(f"-> Dang du doan tren tap eval ({len(data['X_eval'])} mau)...")
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
