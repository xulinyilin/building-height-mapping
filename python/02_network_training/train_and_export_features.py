#!/usr/bin/env python3
"""Train the supplied dual-branch CNN and export final-convolution results."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from model import BranchConfig, DualBranchHeightCNN


LOGGER = logging.getLogger("train_dual_branch_cnn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the height model and export the 32-D final-convolution feature "
            "and trained convolution kernels."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> tuple[dict[str, Any], Path]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    project_root = Path(config.get("project_root", "."))
    if not project_root.is_absolute():
        project_root = (path.parent / project_root).resolve()
    return config, project_root


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        if deterministic:
            torch.use_deterministic_algorithms(True)


class LMDataSet(Dataset):
    """Six-band NPY patches plus a parallel JSON property list."""

    def __init__(
        self,
        np_filename: Path,
        json_filename: Path,
        label_key: str = "Height",
        sample_id_key: str | None = "sample_id",
        year_key: str | None = "year",
        label_scale: float = 1.0,
        rgb_scale: float = 1600.0,
        nir_scale: float = 3400.0,
        radar_min: float = -50.0,
        radar_max: float = 1.0,
    ) -> None:
        if label_scale <= 0 or rgb_scale == 0 or nir_scale == 0:
            raise ValueError("label_scale, rgb_scale and nir_scale must be valid non-zero values")
        if radar_max <= radar_min:
            raise ValueError("radar_max must exceed radar_min")
        self.data = np.load(np_filename, mmap_mode="r", allow_pickle=False)
        with json_filename.open("r", encoding="utf-8") as stream:
            properties = json.load(stream)
        if not isinstance(properties, list):
            raise ValueError("Properties JSON must contain one list of sample objects")
        if len(properties) != len(self.data):
            raise ValueError(
                f"NPY has {len(self.data)} rows but JSON has {len(properties)} records"
            )
        if self.data.ndim != 4 or self.data.shape[1] != 6:
            raise ValueError("Input NPY must have shape [N, 6, H, W]")
        if self.data.shape[2] < 35 or self.data.shape[3] < 35:
            raise ValueError("Every input patch must be at least 35 x 35")
        missing_labels = [index for index, item in enumerate(properties) if label_key not in item]
        if missing_labels:
            raise KeyError(f"Missing JSON label key {label_key!r} at row {missing_labels[0]}")

        self.labels = np.asarray([item[label_key] for item in properties], dtype=np.float32)
        if not np.all(np.isfinite(self.labels)):
            raise ValueError("Height labels contain NaN or infinite values")
        self.sample_ids = [
            str(item.get(sample_id_key, index)) if sample_id_key else str(index)
            for index, item in enumerate(properties)
        ]
        self.years = [
            item.get(year_key, "") if year_key else "" for item in properties
        ]
        self.label_scale = float(label_scale)
        self.rgb_scale = float(rgb_scale)
        self.nir_scale = float(nir_scale)
        self.radar_min = float(radar_min)
        self.radar_range = float(radar_max - radar_min)

        LOGGER.info(
            "%s: n=%d, height mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            json_filename,
            len(self.labels),
            float(np.mean(self.labels)),
            float(np.std(self.labels)),
            float(np.min(self.labels)),
            float(np.max(self.labels)),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patch = np.asarray(self.data[index], dtype=np.float32)
        if not np.all(np.isfinite(patch)):
            raise ValueError(f"Input contains NaN or infinite values at row {index}")
        rgb = patch[:3] / self.rgb_scale
        nir = patch[3:4] / self.nir_scale
        radar = (patch[4:] - self.radar_min) / self.radar_range
        inputs = np.concatenate((rgb, nir, radar), axis=0)
        height = float(self.labels[index])
        return {
            "inputs": torch.from_numpy(inputs),
            "target": torch.tensor(height / self.label_scale, dtype=torch.float32),
            "height": torch.tensor(height, dtype=torch.float32),
            "sample_id": self.sample_ids[index],
            "year": str(self.years[index]),
            "row_index": index,
        }


def branch_config(raw: dict[str, Any]) -> BranchConfig:
    return BranchConfig(
        in_channels=int(raw["in_channels"]),
        channels=tuple(int(value) for value in raw["channels"]),
        kernels=tuple(int(value) for value in raw["kernels"]),
        bias=bool(raw.get("bias", False)),
    )


def dataset_from_config(
    split: str, config: dict[str, Any], root: Path
) -> LMDataSet | None:
    inputs = config["inputs"]
    npy_key = f"{split}_data_npy"
    json_key = f"{split}_properties_json"
    if not inputs.get(npy_key) and not inputs.get(json_key):
        return None
    if not inputs.get(npy_key) or not inputs.get(json_key):
        raise ValueError(f"Both {npy_key} and {json_key} must be supplied")
    npy_path = resolve_path(inputs[npy_key], root)
    json_path = resolve_path(inputs[json_key], root)
    if not npy_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Missing {split} input: {npy_path} or {json_path}")
    label = config.get("label", {})
    normalization = config.get("normalization", {})
    return LMDataSet(
        np_filename=npy_path,
        json_filename=json_path,
        label_key=str(label.get("key", "Height")),
        sample_id_key=label.get("sample_id_key", "sample_id"),
        year_key=label.get("year_key", "year"),
        label_scale=float(label.get("scale", 1.0)),
        rgb_scale=float(normalization.get("rgb_divisor", 1600.0)),
        nir_scale=float(normalization.get("nir_divisor", 3400.0)),
        radar_min=float(normalization.get("radar_min", -50.0)),
        radar_max=float(normalization.get("radar_max", 1.0)),
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=workers > 0,
    )


def regression_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - reference
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2,
    }


def run_epoch(
    model: DualBranchHeightCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_scale: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_scaled_squared_error = 0.0
    count = 0
    references: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True).float()
        targets = batch["target"].to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(inputs)["height"]
            loss = criterion(output, targets)
            if training:
                loss.backward()
                optimizer.step()
        batch_count = targets.shape[0]
        total_scaled_squared_error += float(loss.detach().cpu()) * batch_count
        count += batch_count
        predictions.append(output.detach().cpu().numpy() * label_scale)
        references.append(batch["height"].numpy())

    if count == 0:
        raise ValueError("Data loader is empty")
    values = regression_metrics(np.concatenate(references), np.concatenate(predictions))
    values["scaled_mse_loss"] = total_scaled_squared_error / count
    return values


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_features(
    model: DualBranchHeightCNN,
    named_datasets: Iterable[tuple[str, Dataset]],
    output_path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    label_scale: float,
    seed: int,
) -> None:
    """Export the two final 1 x 1 x 16 convolution outputs as 32 columns."""
    columns = [
        *[f"ms_conv_{index:02d}" for index in range(1, 17)],
        *[f"rd_conv_{index:02d}" for index in range(1, 17)],
    ]
    model.eval()
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            ["sample_id", "year", "subset", "height_label_m", "cnn_height_m", *columns]
        )
        with torch.no_grad():
            for subset_name, dataset in named_datasets:
                loader = make_loader(
                    dataset, batch_size, False, workers, device, seed
                )
                for batch in loader:
                    inputs = batch["inputs"].to(device, non_blocking=True).float()
                    output = model(inputs)
                    features = output["deep_features"].detach().cpu().numpy()
                    predicted = output["height"].detach().cpu().numpy() * label_scale
                    heights = batch["height"].numpy()
                    for row in range(len(heights)):
                        writer.writerow(
                            [
                                batch["sample_id"][row],
                                batch["year"][row],
                                subset_name,
                                f"{heights[row]:.8g}",
                                f"{predicted[row]:.8g}",
                                *[f"{value:.9g}" for value in features[row]],
                            ]
                        )


def convolution_weights(branch: nn.Module) -> list[np.ndarray]:
    return [
        module.weight.detach().float().cpu().numpy()
        for module in branch.modules()
        if isinstance(module, nn.Conv2d)
    ]


def export_gee_branch(
    branch_name: str,
    branch: nn.Module,
    output_path: Path,
    input_code: list[str],
) -> None:
    """Write ee.Kernel.fixed definitions and convNtoN2 calls for one branch."""
    weights = convolution_weights(branch)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("// Trained kernels exported from PyTorch.\n")
        stream.write("// Requires the project's convNtoN2 helper in Google Earth Engine.\n")
        stream.write("// PyTorch cross-correlation convention is preserved by convNtoN2.\n\n")
        for line in input_code:
            stream.write(line + "\n")
        stream.write("\n")
        for layer_index, data in enumerate(weights):
            output_channels, input_channels, rows, columns = data.shape
            for output_index in range(output_channels):
                for input_index in range(input_channels):
                    variable = (
                        f"{branch_name}_{layer_index}_{output_index}_{input_index}"
                    )
                    matrix = json.dumps(data[output_index, input_index].tolist())
                    stream.write(
                        f"var {variable} = ee.Kernel.fixed({rows}, {columns}, {matrix});\n"
                    )
            list_name = f"{branch_name}_ker_list_{layer_index}"
            stream.write(f"var {list_name} = [\n")
            for output_index in range(output_channels):
                variables = ", ".join(
                    f"{branch_name}_{layer_index}_{output_index}_{input_index}"
                    for input_index in range(input_channels)
                )
                stream.write(f"  [{variables}],\n")
            stream.write("];\n\n")

        current_image = f"{branch_name}_input"
        for layer_index, data in enumerate(weights):
            output_channels, input_channels = data.shape[:2]
            output_image = f"{branch_name}_{layer_index}_out_im"
            kernel_list = f"{branch_name}_ker_list_{layer_index}"
            stream.write(
                f"var {output_image} = convNtoN2({current_image}, {kernel_list}, "
                f"{input_channels}, {output_channels}, \"{output_image}\");\n"
            )
            stream.write(
                f"{output_image} = {output_image}.gt(0).multiply({output_image});\n"
            )
            current_image = output_image
        stream.write(f"\n// Final convolution output: {current_image} (16 bands).\n")


def export_flat_parameters(model: nn.Module, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as stream:
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            values = module.weight.detach().float().cpu().numpy()
            stream.write(f"[{name}.weight]\nshape={list(values.shape)}\n")
            flat = values.reshape(-1)
            for start in range(0, len(flat), 12):
                stream.write(" ".join(f"{value:.9g}" for value in flat[start : start + 12]))
                stream.write("\n")
            stream.write("\n")


def ensure_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; use --overwrite: " + ", ".join(map(str, existing)))


def main() -> int:
    args = parse_args()
    config, root = load_yaml(args.config.resolve())
    logging.basicConfig(
        level=getattr(logging, str(config.get("logging", {}).get("level", "INFO")).upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    training = config["training"]
    seed = int(config.get("random_seed", 2023))
    set_reproducibility(seed, bool(training.get("deterministic", True)))
    device_setting = str(training.get("device", "auto"))
    device = torch.device(
        "cuda" if device_setting == "auto" and torch.cuda.is_available()
        else "cpu" if device_setting == "auto"
        else device_setting
    )
    LOGGER.info("Using device: %s", device)

    train_dataset = dataset_from_config("train", config, root)
    if train_dataset is None:
        raise ValueError("Training NPY and properties JSON are required")
    validation_dataset = dataset_from_config("validation", config, root)
    test_dataset = dataset_from_config("test", config, root)

    if validation_dataset is None:
        indices = np.arange(len(train_dataset))
        np.random.default_rng(seed).shuffle(indices)
        split_at = int(float(training.get("fallback_train_fraction", 0.8)) * len(indices))
        if split_at <= 0 or split_at >= len(indices):
            raise ValueError("Fallback split leaves an empty train or validation subset")
        training_data: Dataset = Subset(train_dataset, indices[:split_at].tolist())
        validation_data: Dataset = Subset(train_dataset, indices[split_at:].tolist())
        LOGGER.warning("No validation files supplied; using a seeded row-level 80/20 split")
    else:
        training_data = train_dataset
        validation_data = validation_dataset

    batch_size = int(training.get("batch_size", 64))
    workers = int(training.get("workers", 0))
    train_loader = make_loader(training_data, batch_size, True, workers, device, seed)
    validation_loader = make_loader(
        validation_data, batch_size, False, workers, device, seed
    )

    ms_config = branch_config(config["model"]["ms_branch"])
    rd_config = branch_config(config["model"]["rd_branch"])
    model = DualBranchHeightCNN(
        ms_config=ms_config,
        rd_config=rd_config,
        dropout=float(config["model"].get("dropout", 0.3)),
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    label_scale = float(config.get("label", {}).get("scale", 1.0))

    output_dir = resolve_path(config["outputs"]["directory"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "checkpoint": output_dir / "best_dual_branch_cnn.pt",
        "history": output_dir / "training_history.csv",
        "features": output_dir / "deep_convolution_features.txt",
        "parameters": output_dir / "convolution_parameters.txt",
        "ms_gee": output_dir / "ms_branch_togee.txt",
        "rd_gee": output_dir / "rd_branch_togee.txt",
        "metadata": output_dir / "training_metadata.json",
    }
    ensure_outputs(paths, args.overwrite)

    history: list[dict[str, Any]] = []
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs = int(training.get("epochs", 200))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, label_scale, optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, label_scale
        )
        row: dict[str, Any] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        LOGGER.info(
            "Epoch %03d/%03d | train MSE %.6f R2 %.4f | val MSE %.6f R2 %.4f",
            epoch,
            epochs,
            train_metrics["mse"],
            train_metrics["r2"],
            validation_metrics["mse"],
            validation_metrics["r2"],
        )
        if validation_metrics["mse"] < best_validation_mse:
            best_validation_mse = validation_metrics["mse"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "ms_config": asdict(ms_config),
                    "rd_config": asdict(rd_config),
                    "dropout": float(config["model"].get("dropout", 0.3)),
                    "label_scale": label_scale,
                    "best_epoch": best_epoch,
                },
                paths["checkpoint"],
            )

    write_history(paths["history"], history)
    try:
        checkpoint = torch.load(paths["checkpoint"], map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(paths["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    named_datasets: list[tuple[str, Dataset]] = [
        ("train", training_data),
        ("validation", validation_data),
    ]
    evaluation: dict[str, dict[str, float]] = {
        "train": run_epoch(model, train_loader, criterion, device, label_scale),
        "validation": run_epoch(model, validation_loader, criterion, device, label_scale),
    }
    if test_dataset is not None:
        test_loader = make_loader(test_dataset, batch_size, False, workers, device, seed)
        named_datasets.append(("test", test_dataset))
        evaluation["test"] = run_epoch(model, test_loader, criterion, device, label_scale)

    export_features(
        model,
        named_datasets,
        paths["features"],
        device,
        batch_size,
        workers,
        label_scale,
        seed,
    )
    export_flat_parameters(model, paths["parameters"])
    gee = config.get("gee", {})
    planet_image = str(gee.get("multispectral_image", "planet_im"))
    radar_image = str(gee.get("radar_image", "radar_im"))
    ms_band_list = list(gee.get("multispectral_bands", ["R", "G", "B", "NIR"]))
    rd_band_list = list(gee.get("radar_bands", ["VV", "VH"]))
    if len(ms_band_list) != 4 or len(rd_band_list) != 2:
        raise ValueError("GEE export requires four multispectral and two radar band names")
    normalization = config.get("normalization", {})
    rgb_divisor = float(normalization.get("rgb_divisor", 1600.0))
    nir_divisor = float(normalization.get("nir_divisor", 3400.0))
    radar_min = float(normalization.get("radar_min", -50.0))
    radar_range = float(normalization.get("radar_max", 1.0)) - radar_min
    export_gee_branch(
        "ms_branch",
        model.ms_branch,
        paths["ms_gee"],
        [
            f"var ms_rgb = {planet_image}.select({json.dumps(ms_band_list[:3])})"
            f".divide({rgb_divisor});",
            f"var ms_nir = {planet_image}.select({json.dumps(ms_band_list[3:4])})"
            f".divide({nir_divisor});",
            "var ms_branch_input = ee.Image.cat([ms_rgb, ms_nir]);",
        ],
    )
    export_gee_branch(
        "rd_branch",
        model.rd_branch,
        paths["rd_gee"],
        [
            f"var rd_branch_input = {radar_image}.select({json.dumps(rd_band_list)})"
            f".subtract({radar_min}).divide({radar_range});"
        ],
    )

    metadata = {
        "architecture": {
            "ms_branch": asdict(ms_config),
            "rd_branch": asdict(rd_config),
            "padding": 0,
            "pooling": None,
            "batch_normalization": False,
            "final_convolution_feature_dimension": 32,
        },
        "best_epoch": best_epoch,
        "evaluation_metres": evaluation,
        "normalization": {
            "rgb_divisor": rgb_divisor,
            "nir_divisor": nir_divisor,
            "radar_min": radar_min,
            "radar_max": radar_min + radar_range,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    with paths["metadata"].open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    LOGGER.info("Best epoch: %d; convolution features: %s", best_epoch, paths["features"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
