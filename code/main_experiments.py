from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import argparse
import csv
import random
import time

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.baseline_vit import build_baseline_vit
from models.performer_vit import build_performer_vit
from models.circulant_string_vit import build_circulant_string_vit
from models.cayley_string_vit import build_cayley_string_vit
from plot_results import make_all_plots


# Part 2: Global Experiment Config
@dataclass(frozen=True)
class ExperimentConfig:
    # Data settings
    data_dir: str = "./data"
    output_dir: str = "../results"

    # Main experiment design
    datasets: Tuple[str, ...] = ("cifar10", "cifar100", "mnist", "fashion_mnist")
    models: Tuple[str, ...] = ("baseline", "performer", "circulant_string", "cayley_string")
    seeds: Tuple[int, ...] = (0, 1, 2)

    # Training settings
    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    num_workers: int = 2
    train_size: int = 10000
    val_size: int = 2000
    device: str = "auto"

    # ViT backbone settings
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1

    # Performer settings
    performer_features: int = 64
    performer_redraw_projection: bool = False

    # Logging
    raw_results_file: str = "results.csv"
    summary_file: str = "results_summary_mean_std.csv"
    latex_file: str = "results_summary_latex_table.txt"


DATASET_INFO: Dict[str, Dict[str, int | str]] = {
    "cifar10": {"num_classes": 10, "display_name": "CIFAR-10"},
    "cifar100": {"num_classes": 100, "display_name": "CIFAR-100"},
    "mnist": {"num_classes": 10, "display_name": "MNIST"},
    "fashion_mnist": {"num_classes": 10, "display_name": "Fashion-MNIST"},
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "baseline": "Baseline",
    "performer": "Performer",
    "circulant_string": "Circulant-STRING",
    "cayley_string": "Cayley-STRING",
}



# Part 3: Reproducibility and Device Helper Functions
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    Set random seeds to make the experiment as reproducible as possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These settings improve reproducibility, but may slightly reduce speed.
    # These settings improve reproducibility, but may slightly reduce speed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested_device: str) -> torch.device:
    """Resolve runtime device.
    Resolve the current runtime device.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(requested_device)


def sync_if_needed(device_obj: torch.device) -> None:
    """Synchronise GPU timing when necessary.
    Synchronize GPU timing to avoid asynchronous execution affecting training/inference time.
    """
    if device_obj.type == "cuda":
        torch.cuda.synchronize()


def print_device_info(device: torch.device) -> None:
    """Print basic device information.
    Print device information.
    """
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))



# Part 4: Datasets and DataLoaders
def build_transforms(dataset_name: str, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train/eval transforms.
    Build transforms for the training and validation sets.
    """
    normalize = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    if dataset_name in {"mnist", "fashion_mnist"}:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])
        return transform, transform

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return transform, transform


def load_dataset(dataset_name: str, data_dir: str, train: bool, transform):
    """Load one torchvision dataset.
    Load a torchvision dataset.
    """
    name = dataset_name.lower()

    if name == "cifar10":
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)

    if name == "cifar100":
        return datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)

    if name == "mnist":
        return datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)

    if name == "fashion_mnist":
        return datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_subset(dataset, subset_size: int, seed: int) -> Subset:
    """Create a deterministic random subset.
    Create a deterministic random subset so different models use the same samples under the same seed.
    """
    subset_n = min(subset_size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_n].tolist()
    return Subset(dataset, indices)


def build_dataloaders(
    dataset_name: str,
    config: ExperimentConfig,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders.
    Build DataLoaders for the training and validation sets.
    """
    train_transform, eval_transform = build_transforms(dataset_name, config.image_size)

    train_dataset_full = load_dataset(
        dataset_name=dataset_name,
        data_dir=config.data_dir,
        train=True,
        transform=train_transform,
    )
    val_dataset_full = load_dataset(
        dataset_name=dataset_name,
        data_dir=config.data_dir,
        train=False,
        transform=eval_transform,
    )

    train_dataset = make_subset(train_dataset_full, config.train_size, seed)
    val_dataset = make_subset(val_dataset_full, config.val_size, seed + 10000)

    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader



def build_model(model_type: str, dataset_name: str, config: ExperimentConfig) -> nn.Module:
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_classes = int(DATASET_INFO[dataset_name]["num_classes"])
    builders = {
        "baseline": build_baseline_vit,
        "performer": build_performer_vit,
        "circulant_string": build_circulant_string_vit,
        "cayley_string": build_cayley_string_vit,
    }
    if model_type not in builders:
        raise ValueError(f"Unsupported model: {model_type}")
    return builders[model_type](num_classes=num_classes, config=config)


# Part 7: Training and Evaluation
def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int,
) -> float:
    """Train one model and return wall-clock training time.
    Train one model and return the training time.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sync_if_needed(device)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        print(f"    Epoch {epoch + 1}/{epochs}")

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"      batch {batch_idx:04d}/{total_batches:04d} | loss = {loss.item():.4f}")

        avg_loss = running_loss / max(total_batches, 1)
        print(f"      avg train loss = {avg_loss:.4f}")

    sync_if_needed(device)
    end_time = time.time()

    return end_time - start_time


def evaluate_with_timing(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate accuracy and inference time.
    Evaluate accuracy and inference time.
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    sync_if_needed(device)
    start_time = time.time()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    sync_if_needed(device)
    end_time = time.time()

    accuracy = correct / total if total > 0 else 0.0
    inference_time = end_time - start_time
    return accuracy, inference_time


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters.
    Count the number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Part 8: CSV Logging
def append_result_row(raw_results_path: Path, row: Dict[str, object]) -> None:
    """Append one row to raw results CSV.
    Append one row to the raw results CSV.
    """
    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = raw_results_path.is_file()

    with raw_results_path.open(mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_raw_results(raw_results_path: Path) -> List[Dict[str, str]]:
    """Read raw results CSV.
    Read the raw results CSV.
    """
    if not raw_results_path.exists():
        raise FileNotFoundError(f"No raw results file found: {raw_results_path}")

    with raw_results_path.open(mode="r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """Write rows to CSV.
    Write to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



# Part 9: Result Aggregation
def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    """Return sample mean and sample std.
    Return the mean and sample standard deviation.
    """
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate_results(config: ExperimentConfig) -> List[Dict[str, object]]:
    """Aggregate raw results into mean ± std summaries.
    Aggregate raw results as mean ± std.
    """
    output_dir = Path(config.output_dir)
    raw_path = output_dir / config.raw_results_file
    summary_path = output_dir / config.summary_file
    latex_path = output_dir / config.latex_file

    rows = read_raw_results(raw_path)

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset"], row["model"])
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []

    for dataset_name in config.datasets:
        for model_name in config.models:
            key = (dataset_name, model_name)
            group = grouped.get(key, [])

            acc_mean, acc_std = mean_std(float(r["accuracy"]) for r in group)
            train_mean, train_std = mean_std(float(r["training_time"]) for r in group)
            infer_mean, infer_std = mean_std(float(r["inference_time"]) for r in group)

            summary_rows.append({
                "dataset": dataset_name,
                "dataset_display": DATASET_INFO[dataset_name]["display_name"],
                "model": model_name,
                "model_display": MODEL_DISPLAY_NAMES[model_name],
                "n_runs": len(group),
                "accuracy_mean": f"{acc_mean:.6f}",
                "accuracy_std": f"{acc_std:.6f}",
                "training_time_mean": f"{train_mean:.6f}",
                "training_time_std": f"{train_std:.6f}",
                "inference_time_mean": f"{infer_mean:.6f}",
                "inference_time_std": f"{infer_std:.6f}",
            })

    fieldnames = [
        "dataset",
        "dataset_display",
        "model",
        "model_display",
        "n_runs",
        "accuracy_mean",
        "accuracy_std",
        "training_time_mean",
        "training_time_std",
        "inference_time_mean",
        "inference_time_std",
    ]

    write_csv(summary_path, summary_rows, fieldnames)
    write_latex_accuracy_table(summary_rows, latex_path)

    print(f"\nSummary saved to: {summary_path}")
    print(f"LaTeX table saved to: {latex_path}")

    return summary_rows


def write_latex_accuracy_table(summary_rows: List[Dict[str, object]], latex_path: Path) -> None:
    """Write a LaTeX-ready accuracy table.
    Write a LaTeX accuracy table that can be directly pasted into the paper.
    """
    datasets_order = ["cifar10", "cifar100", "mnist", "fashion_mnist"]
    models_order = ["baseline", "performer", "circulant_string", "cayley_string"]

    lookup = {
        (str(r["dataset"]), str(r["model"])): r
        for r in summary_rows
    }

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main validation accuracy results averaged over multiple random seeds.}")
    lines.append(r"\label{tab:main_accuracy_results}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Model & CIFAR-10 & CIFAR-100 & MNIST & Fashion-MNIST \\")
    lines.append(r"\hline")

    for model_name in models_order:
        model_display = MODEL_DISPLAY_NAMES[model_name]
        cells = []
        for dataset_name in datasets_order:
            row = lookup.get((dataset_name, model_name))
            if row is None:
                cells.append("--")
            else:
                mean_val = float(row["accuracy_mean"])
                std_val = float(row["accuracy_std"])
                cells.append(f"{mean_val:.4f} $\\pm$ {std_val:.4f}")
        lines.append(f"{model_display} & " + " & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(lines), encoding="utf-8")



# Part 11: Single Run and Full Experiment Loop
def run_single_experiment(
    dataset_name: str,
    model_name: str,
    seed: int,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    """Run one dataset-model-seed experiment.
    Run one dataset-model-seed combination experiment.
    """
    print("\n" + "=" * 80)
    print(f"Dataset: {DATASET_INFO[dataset_name]['display_name']} | "
          f"Model: {MODEL_DISPLAY_NAMES[model_name]} | Seed: {seed}")
    print("=" * 80)

    set_seed(seed)

    train_loader, val_loader = build_dataloaders(
        dataset_name=dataset_name,
        config=config,
        seed=seed,
    )

    model = build_model(
        model_type=model_name,
        dataset_name=dataset_name,
        config=config,
    )

    num_params = count_trainable_parameters(model)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Trainable parameters: {num_params}")

    # Quick forward sanity check.
    # Quick forward sanity check.
    dummy_x = torch.randn(
        2,
        config.in_channels,
        config.image_size,
        config.image_size,
        device=device,
    )
    model.to(device)
    with torch.no_grad():
        dummy_out = model(dummy_x)
    print("Dummy output shape:", tuple(dummy_out.shape))

    training_time = train_one_model(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=config.lr,
        epochs=config.epochs,
    )

    accuracy, inference_time = evaluate_with_timing(
        model=model,
        val_loader=val_loader,
        device=device,
    )

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "dataset_display": DATASET_INFO[dataset_name]["display_name"],
        "model": model_name,
        "model_display": MODEL_DISPLAY_NAMES[model_name],
        "seed": seed,
        "accuracy": f"{accuracy:.6f}",
        "training_time": f"{training_time:.6f}",
        "inference_time": f"{inference_time:.6f}",
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "train_size": config.train_size,
        "val_size": config.val_size,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "embed_dim": config.embed_dim,
        "depth": config.depth,
        "num_heads": config.num_heads,
        "mlp_ratio": config.mlp_ratio,
        "dropout": config.dropout,
        "performer_features": config.performer_features,
        "num_trainable_parameters": num_params,
        "device": str(device),
    }

    print("\nRun result:")
    print(f"  accuracy       = {accuracy:.4f}")
    print(f"  training_time  = {training_time:.2f}s")
    print(f"  inference_time = {inference_time:.2f}s")

    return row


def run_all_experiments(config: ExperimentConfig) -> None:
    """Run all dataset-model-seed experiments.
    Run all dataset-model-seed combination experiments.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_dir / config.raw_results_file

    device = resolve_device(config.device)
    print_device_info(device)

    print("\nExperiment plan:")
    print("  Datasets:", ", ".join(config.datasets))
    print("  Models:", ", ".join(config.models))
    print("  Seeds:", ", ".join(str(s) for s in config.seeds))
    print("  Total runs:", len(config.datasets) * len(config.models) * len(config.seeds))
    print("  Epochs:", config.epochs)
    print("  Train subset size:", config.train_size)
    print("  Validation subset size:", config.val_size)
    print("  Output directory:", output_dir)

    for dataset_name in config.datasets:
        for model_name in config.models:
            for seed in config.seeds:
                row = run_single_experiment(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    config=config,
                    device=device,
                )
                append_result_row(raw_results_path, row)
                print(f"Saved raw result to: {raw_results_path}")

    print("\nAll runs completed.")
    aggregate_results(config)
    make_all_plots(config, DATASET_INFO, MODEL_DISPLAY_NAMES)



# Part 12: Command-Line Interface
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main experiments for Efficient Vision Transformer Models."
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="../results")

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["cifar10", "cifar100", "mnist", "fashion_mnist"],
        choices=["cifar10", "cifar100", "mnist", "fashion_mnist"],
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["baseline", "performer", "circulant_string", "cayley_string"],
        choices=["baseline", "performer", "circulant_string", "cayley_string"],
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate and plot existing raw results.",
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=args.output_dir,
        datasets=tuple(args.datasets),
        models=tuple(args.models),
        seeds=tuple(args.seeds),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        train_size=args.train_size,
        val_size=args.val_size,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)

    if args.aggregate_only:
        aggregate_results(config)
        make_all_plots(config, DATASET_INFO, MODEL_DISPLAY_NAMES)
    else:
        run_all_experiments(config)


# In a notebook, do not auto-run argparse-based CLI.
# Use the notebook-friendly cells below instead.
if __name__ == "__main__":
    main()

# Full experiment configuration for notebook use
# You can reduce datasets/seeds/epochs here for a quick test.
notebook_config = ExperimentConfig(
    output_dir="../results",
    datasets=("cifar10", "cifar100", "mnist", "fashion_mnist"),
    models=("baseline", "performer", "circulant_string", "cayley_string"),
    seeds=(0, 1, 2),
    epochs=5,
    train_size=10000,
    val_size=2000,
    batch_size=32,
    device="auto",
)

# Uncomment to run all experiments.
# run_all_experiments(notebook_config)

# If you already have results.csv and only want to regenerate summary tables and figures:
# aggregate_results(notebook_config)
# make_all_plots(notebook_config, DATASET_INFO, MODEL_DISPLAY_NAMES)

