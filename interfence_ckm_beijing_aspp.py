"""Evaluate ASPP Flow Matching on CKMImageNet-Beijing.

This mirrors the existing ASPP/interfence_aspp.py role while using the
CKMImageNet-Beijing dataset wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

ASPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ASPP_DIR.parent
CKM_BEIJING_DIR = PROJECT_ROOT / "ckm_beijing"

if str(ASPP_DIR) not in sys.path:
    sys.path.insert(0, str(ASPP_DIR))
if str(CKM_BEIJING_DIR) not in sys.path:
    sys.path.insert(0, str(CKM_BEIJING_DIR))

from fm_aspp import FlowMatchingModel  
from dataset import CKMImageNetBeijingDataset, compute_metrics, load_split, parse_environment_source, to_zero_one  


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def save_panel_jpg(path: Path, image: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6), dpi=100)
    plt.imshow(image, cmap=cmap_name, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(path, format="jpg", dpi=100, facecolor="white")
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device, target_range: str, metric_mode: str, steps: int, solver: str, output_dir: Path, save_visuals: int) -> dict:
    model.eval()
    rows = []
    times = []
    visual_dir = output_dir / "visualizations"
    if save_visuals > 0:
        visual_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for batch in tqdm(loader, desc="eval"):
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["gain_map"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = model.sample(condition, steps=steps, device=device, solver=solver)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        pred_01 = to_zero_one(pred.detach().cpu(), target_range)
        target_01 = to_zero_one(target.detach().cpu(), target_range)
        per_sample_ms = elapsed * 1000.0 / max(condition.size(0), 1)

        for i in range(pred_01.size(0)):
            pred_np = pred_01[i, 0].numpy()
            target_np = target_01[i, 0].numpy()
            metric = compute_metrics(pred_np, target_np, metric_mode)
            metric.update(
                {
                    "bs_id": int(batch["bs_id"][i]),
                    "row": int(batch["row"][i]),
                    "col": int(batch["col"][i]),
                    "inference_time_ms": per_sample_ms,
                }
            )
            rows.append(metric)
            times.append(per_sample_ms)
            if saved < save_visuals:
                stem = f"bs{metric['bs_id']}_r{metric['row']}_c{metric['col']}"
                error = np.abs(pred_np - target_np)
                np.save(visual_dir / f"{stem}_aspp.npy", pred_np)
                np.save(visual_dir / f"{stem}_gt.npy", target_np)
                save_panel_jpg(visual_dir / f"{stem}_aspp.jpg", pred_np, "viridis", 0.0, 1.0)
                save_panel_jpg(visual_dir / f"{stem}_gt.jpg", target_np, "viridis", 0.0, 1.0)
                save_panel_jpg(visual_dir / f"{stem}_abs_error.jpg", error, "magma", 0.0, float(max(error.max(), 1e-6)))
                saved += 1

    summary = {key: float(np.mean([row[key] for row in rows])) for key in ["mse", "rmse", "nmse", "psnr", "ssim"]}
    summary["inference_time_ms"] = float(np.mean(times))
    summary["num_samples"] = len(rows)
    summary["steps"] = steps
    summary["solver"] = solver
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "test_metrics_per_sample.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beijing_dir", type=Path, default=Path(r"D:\CKM\CKMImageNet-main\CKMImageNet-main\Beijing"))
    parser.add_argument("--split_file", type=Path, default=PROJECT_ROOT / "ckm_beijing" / "splits_beijing.json")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "ckm_beijing" / "outputs" / "aspp_targetmask" / "aspp_fm_best.pth")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "ckm_beijing" / "outputs" / "aspp_targetmask")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--condition_range", choices=["zero_one", "minus_one_one"], default="minus_one_one")
    parser.add_argument("--target_range", choices=["zero_one", "minus_one_one"], default="minus_one_one")
    parser.add_argument("--environment_source", type=parse_environment_source, default=None)
    parser.add_argument("--metric_mode", choices=["paper_legacy", "zero_one"], default="paper_legacy")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--save_visuals", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    hidden_dim = int(ckpt_args.get("hidden_dim", 64))
    num_layers = int(ckpt_args.get("num_layers", 4))
    environment_source = args.environment_source or ckpt_args.get("environment_source", "target_mask")

    _, _, test_samples = load_split(args.split_file)
    test_ds = CKMImageNetBeijingDataset(
        beijing_dir=args.beijing_dir,
        samples=test_samples,
        condition_range=args.condition_range,
        target_range=args.target_range,
        environment_source=environment_source,
    )
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers)

    model = FlowMatchingModel(condition_dim=2, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        target_range=args.target_range,
        metric_mode=args.metric_mode,
        steps=args.steps,
        solver=args.solver,
        output_dir=args.output_dir,
        save_visuals=args.save_visuals,
    )
    print(
        f"Test | NMSE={metrics['nmse']:.6f} | RMSE={metrics['rmse']:.6f} | "
        f"SSIM={metrics['ssim']:.4f} | PSNR={metrics['psnr']:.2f} | "
        f"time={metrics['inference_time_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
