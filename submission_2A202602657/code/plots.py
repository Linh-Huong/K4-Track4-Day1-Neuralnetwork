"""plots.py — Vẽ đồ thị cho từng thí nghiệm và đồ thị so sánh nhóm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    hist = result["history"]
    summary = result.get("summary", {})
    epochs = hist["epoch"]
    best_epoch = summary.get("best_epoch", None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    exp_id = cfg.get("exp_id", "run")
    opt = cfg.get("optimizer", "")
    lr = cfg.get("lr", "")
    batch = cfg.get("batch", "")
    fig.suptitle(f"{exp_id} | opt={opt}, lr={lr}, batch={batch}, loss={cfg.get('loss', 'ce')}",
                 fontsize=13, fontweight="bold")

    # (1) Train & Val Loss
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", marker="o", markersize=3)
    ax.plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", marker="s", markersize=3)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax.set_title("Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    # (2) Val Accuracy & Macro-F1
    ax = axes[1]
    ax.plot(epochs, hist["val_acc"], label="Val Acc", color="#2ca02c", marker="^", markersize=3)
    ax.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", marker="d", markersize=3)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax.set_title("Metrics vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    # (3) Grad Norm
    ax = axes[2]
    ax.plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="#9467bd", marker="x", markersize=3)
    ax.set_title("Gradient Norm vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"-> Da luu bieu do vao: {p}")


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "")
        if metric in hist:
            ax.plot(hist["epoch"], hist[metric], label=exp_id, marker="o", markersize=3)

    display_title = title if title else f"So sanh {metric}"
    ax.set_title(display_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"-> Da luu bieu do so sanh vao: {p}")
