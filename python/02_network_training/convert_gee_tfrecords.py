#!/usr/bin/env python3
"""Convert GEE table TFRecords into NPY and JSON files used by the CNN."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from tfrecord.reader import tfrecord_loader


BANDS = ("R", "G", "B", "NIR", "VV", "VH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="GEE TFRecord files or glob patterns.",
    )
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-properties", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=35)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(item) for item in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            candidate = Path(pattern)
            if candidate.is_file():
                paths.append(candidate)

    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise FileNotFoundError("No TFRecord files matched --inputs")
    return unique_paths


def scalar(record: dict[str, np.ndarray], key: str) -> float:
    value = np.asarray(record[key]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{key} must contain one value, found {value.size}")
    return float(value[0])


def main() -> None:
    args = parse_args()
    input_paths = resolve_inputs(args.inputs)

    for output_path in (args.output_data, args.output_properties):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists; use --overwrite to replace it"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

    description = {
        **{band: "float" for band in BANDS},
        "Height": "float",
        "year": "float",
    }
    expected_values = args.patch_size * args.patch_size
    data: list[np.ndarray] = []
    properties: list[dict[str, float | int]] = []

    for input_path in input_paths:
        records = tfrecord_loader(
            data_path=str(input_path),
            index_path=None,
            description=description,
            compression_type="gzip",
        )

        for record in records:
            bands: list[np.ndarray] = []
            for band in BANDS:
                values = np.asarray(record[band], dtype=np.float32).reshape(-1)
                if values.size != expected_values:
                    raise ValueError(
                        f"{input_path}: {band} has {values.size} values; "
                        f"expected {expected_values}"
                    )
                bands.append(
                    values.reshape(args.patch_size, args.patch_size)
                )

            data.append(np.stack(bands, axis=0))
            properties.append(
                {
                    "sample_id": len(properties),
                    "Height": scalar(record, "Height"),
                    "year": int(round(scalar(record, "year"))),
                }
            )

    if not data:
        raise ValueError("The TFRecord files contained no samples")

    output_array = np.stack(data, axis=0).astype(np.float32, copy=False)
    np.save(args.output_data, output_array)
    with args.output_properties.open("w", encoding="utf-8") as stream:
        json.dump(properties, stream, indent=2, ensure_ascii=False)

    print(f"Saved {len(properties)} samples")
    print(f"Data shape: {output_array.shape}")
    print(f"NPY: {args.output_data}")
    print(f"JSON: {args.output_properties}")


if __name__ == "__main__":
    main()
