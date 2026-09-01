from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import CRS, Transformer
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from shapely.geometry import Point


LOGGER = logging.getLogger("prepare_training_samples")


@dataclass(frozen=True)
class RasterSpec:
    path: Path
    bands: tuple[int, ...] = (1,)
    scale: float = 1.0
    offset: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare high-confidence, cross-year stable height samples."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("The configuration root must be a YAML mapping.")

    configured_root = Path(cfg.get("project_root", "."))
    if not configured_root.is_absolute():
        configured_root = (path.parent / configured_root).resolve()
    return cfg, configured_root


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def raster_spec(raw: dict[str, Any], root: Path, default_bands: Sequence[int] = (1,)) -> RasterSpec:
    spec = RasterSpec(
        path=resolve_path(raw["path"], root),
        bands=tuple(int(v) for v in raw.get("bands", default_bands)),
        scale=float(raw.get("scale", 1.0)),
        offset=float(raw.get("offset", 0.0)),
    )
    require_file(spec.path, "Raster")
    return spec


def coordinates_in_raster_crs(points: gpd.GeoDataFrame, raster_crs: Any) -> list[tuple[float, float]]:
    if points.crs is None:
        raise ValueError("Input points have no CRS.")
    source = CRS.from_user_input(points.crs)
    target = CRS.from_user_input(raster_crs)
    if source == target:
        return [(geom.x, geom.y) for geom in points.geometry]
    transformer = Transformer.from_crs(source, target, always_xy=True)
    return [transformer.transform(geom.x, geom.y) for geom in points.geometry]


def sample_raster(points: gpd.GeoDataFrame, spec: RasterSpec) -> np.ndarray:
    """Sample one or more bands; invalid and nodata values become NaN."""
    with rasterio.open(spec.path) as src:
        for band in spec.bands:
            if band < 1 or band > src.count:
                raise ValueError(f"Band {band} is invalid for {spec.path} (count={src.count}).")
        coords = coordinates_in_raster_crs(points, src.crs)
        result = np.full((len(points), len(spec.bands)), np.nan, dtype=np.float64)
        for i, values in enumerate(src.sample(coords, indexes=list(spec.bands), masked=True)):
            array = np.ma.asarray(values, dtype=np.float64)
            if np.ma.is_masked(array) and np.any(np.ma.getmaskarray(array)):
                continue
            result[i, :] = np.asarray(array) * spec.scale + spec.offset
    return result


def relative_height_difference(
    h1: np.ndarray | pd.Series,
    h2: np.ndarray | pd.Series,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Calculate RD = abs((H1 - H2) / H2), matching the QGIS formula.

    H2 is the denominator. Non-finite values and H2 values effectively equal
    to zero return NaN and are excluded by the subsequent sample filters.
    """
    first = np.asarray(h1, dtype=np.float64)
    second = np.asarray(h2, dtype=np.float64)
    result = np.full(np.broadcast(first, second).shape, np.nan, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(second) > epsilon)
    np.divide(
        np.abs(first - second),
        np.abs(second),
        out=result,
        where=valid,
    )
    return result


def _metric_pixel_size(src: rasterio.io.DatasetReader) -> tuple[float, float] | None:
    crs = CRS.from_user_input(src.crs)
    if not crs.is_projected:
        return None
    factors = [axis.unit_conversion_factor for axis in crs.axis_info[:2]]
    if len(factors) < 2 or any(factor is None for factor in factors):
        return None
    return abs(src.res[0]) * factors[0], abs(src.res[1]) * factors[1]


def core_impervious_mask(
    points: gpd.GeoDataFrame,
    mask_path: Path,
    allowed_values: set[float],
    radius_m: float,
    radius_pixels: int | None,
) -> np.ndarray:
    """Check whether every pixel in a circular neighborhood is impervious."""
    keep = np.zeros(len(points), dtype=bool)
    with rasterio.open(mask_path) as src:
        coords = coordinates_in_raster_crs(points, src.crs)
        metric_size = _metric_pixel_size(src)
        if metric_size is None:
            if radius_pixels is None:
                raise ValueError(
                    "ICPIS is not in a projected metric CRS. Set core_radius_pixels explicitly."
                )
            rx = ry = int(radius_pixels)
            pixel_x = pixel_y = 1.0
            radius_units = float(radius_pixels)
        else:
            pixel_x, pixel_y = metric_size
            rx = int(math.ceil(radius_m / pixel_x))
            ry = int(math.ceil(radius_m / pixel_y))
            radius_units = radius_m

        for idx, (x_coord, y_coord) in enumerate(coords):
            row, col = rowcol(src.transform, x_coord, y_coord)
            window = Window(col - rx, row - ry, 2 * rx + 1, 2 * ry + 1)
            data = src.read(1, window=window, boundless=True, masked=True)
            if data.shape != (2 * ry + 1, 2 * rx + 1):
                continue
            yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
            if metric_size is None:
                disk = (xx**2 + yy**2) <= radius_units**2
            else:
                disk = (xx * pixel_x) ** 2 + (yy * pixel_y) ** 2 <= radius_units**2
            values = np.ma.asarray(data)[disk]
            if np.any(np.ma.getmaskarray(values)):
                continue
            keep[idx] = np.all(np.isin(np.asarray(values, dtype=float), list(allowed_values)))
    return keep


def planet_spectral_sd(
    points: gpd.GeoDataFrame,
    annual_specs: Sequence[RasterSpec],
    rgb_weights: Sequence[float],
    ddof: int,
    minimum_valid_years: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the GEE workflow: Gray=0.3R+0.59G+0.11B, then SD across years."""
    cube = np.stack([sample_raster(points, spec) for spec in annual_specs], axis=1)
    weights = np.asarray(rgb_weights, dtype=np.float64)
    if cube.shape[2] != len(weights):
        raise ValueError(
            f"Planet RGB band count ({cube.shape[2]}) does not match "
            f"the number of gray weights ({len(weights)})."
        )
    valid_rgb = np.all(np.isfinite(cube), axis=2)
    gray = np.sum(cube * weights.reshape(1, 1, -1), axis=2)
    gray[~valid_rgb] = np.nan
    valid_years = np.sum(np.isfinite(gray), axis=1)
    if np.any(valid_years <= ddof):
        LOGGER.warning("Some points do not have enough annual grayscale values for SD.")
    spectral_sd = np.nanstd(gray, axis=1, ddof=ddof)
    spectral_sd[valid_years < minimum_valid_years] = np.nan
    return spectral_sd, gray, valid_years


def uniformly_sample_low_rise_pixels(
    base_spec: RasterSpec,
    target_pool_size: int,
    minimum_height: float,
    maximum_height: float,
    output_crs: str,
    seed: int,
) -> gpd.GeoDataFrame:
    """Uniformly sample pixel centers with low-rise values using random-key reservoir sampling."""
    rng = np.random.default_rng(seed)
    selected_keys = np.empty(0, dtype=np.float64)
    selected_rows = np.empty(0, dtype=np.int64)
    selected_cols = np.empty(0, dtype=np.int64)

    with rasterio.open(base_spec.path) as src:
        band = base_spec.bands[0]
        for _, window in src.block_windows(band):
            data = src.read(band, window=window, masked=True).astype(np.float64)
            data = data * base_spec.scale + base_spec.offset
            mask = (
                ~np.ma.getmaskarray(data)
                & np.isfinite(np.asarray(data))
                & (np.asarray(data) > minimum_height)
                & (np.asarray(data) < maximum_height)
            )
            local_rows, local_cols = np.where(mask)
            if local_rows.size == 0:
                continue
            keys = rng.random(local_rows.size)
            global_rows = local_rows + int(window.row_off)
            global_cols = local_cols + int(window.col_off)
            selected_keys = np.concatenate([selected_keys, keys])
            selected_rows = np.concatenate([selected_rows, global_rows])
            selected_cols = np.concatenate([selected_cols, global_cols])
            if selected_keys.size > 2 * target_pool_size:
                chosen = np.argpartition(selected_keys, target_pool_size - 1)[:target_pool_size]
                selected_keys = selected_keys[chosen]
                selected_rows = selected_rows[chosen]
                selected_cols = selected_cols[chosen]

        if selected_keys.size > target_pool_size:
            chosen = np.argpartition(selected_keys, target_pool_size - 1)[:target_pool_size]
            selected_rows = selected_rows[chosen]
            selected_cols = selected_cols[chosen]

        xs, ys = xy(src.transform, selected_rows, selected_cols, offset="center")
        candidates = gpd.GeoDataFrame(
            {"candidate_id": np.arange(len(xs), dtype=np.int64)},
            geometry=[Point(x_coord, y_coord) for x_coord, y_coord in zip(xs, ys)],
            crs=src.crs,
        )
    return candidates.to_crs(output_crs)


def load_candidate_points(
    cfg: dict[str, Any], root: Path, output_crs: str, glo_spec: RasterSpec, seed: int
) -> gpd.GeoDataFrame:
    low_cfg = cfg["sample_selection"]["low_rise"]
    candidate_path = low_cfg.get("candidate_points")
    if candidate_path:
        path = resolve_path(candidate_path, root)
        require_file(path, "Low-rise candidate points")
        points = gpd.read_file(path)
        if points.crs is None:
            raise ValueError("Low-rise candidate points have no CRS.")
        if not np.all(points.geometry.geom_type == "Point"):
            raise ValueError("Low-rise candidate geometries must all be points.")
        return points.to_crs(output_crs)

    target = int(low_cfg["target_count"])
    multiplier = int(low_cfg.get("candidate_multiplier", 10))
    return uniformly_sample_low_rise_pixels(
        base_spec=glo_spec,
        target_pool_size=target * multiplier,
        minimum_height=float(low_cfg.get("minimum_height_m", 0.0)),
        maximum_height=float(low_cfg.get("maximum_height_m", 5.0)),
        output_crs=output_crs,
        seed=seed,
    )


def attach_reference_heights(
    points: gpd.GeoDataFrame, glo_spec: RasterSpec, g3due_spec: RasterSpec
) -> gpd.GeoDataFrame:
    result = points.copy()
    result["glo_m"] = sample_raster(result, glo_spec)[:, 0]
    result["g3due_m"] = sample_raster(result, g3due_spec)[:, 0]
    return result


def apply_gedi_quality_control(
    points: gpd.GeoDataFrame,
    qc_cfg: dict[str, Any],
    summary: dict[str, int],
) -> gpd.GeoDataFrame:
    """Apply the four GEDI L2A quality criteria reported in the manuscript."""
    numeric_fields = [
        "gedi_rh98",
        "quality_flag",
        "degrade_flag",
        "num_detectedmodes",
        "sensitivity",
    ]
    result = points.copy()
    for field in numeric_fields:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    result = result[np.isfinite(result[numeric_fields]).all(axis=1)].copy()
    summary["gedi_after_valid_qc_fields"] = int(len(result))

    result = result[result["quality_flag"] == int(qc_cfg.get("quality_flag", 1))].copy()
    summary["gedi_after_quality_flag"] = int(len(result))

    result = result[result["degrade_flag"] == int(qc_cfg.get("degrade_flag", 0))].copy()
    summary["gedi_after_degrade_flag"] = int(len(result))

    result = result[
        result["num_detectedmodes"]
        > float(qc_cfg.get("num_detectedmodes_greater_than", 1))
    ].copy()
    summary["gedi_after_detected_modes"] = int(len(result))

    result = result[
        result["sensitivity"] > float(qc_cfg.get("sensitivity_greater_than", 0.9))
    ].copy()
    summary["gedi_after_sensitivity"] = int(len(result))
    return result


def prepare_tall_samples(
    cfg: dict[str, Any], root: Path, output_crs: str, glo_spec: RasterSpec, g3due_spec: RasterSpec
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    gedi_cfg = cfg["inputs"]["gedi"]
    gedi_path = resolve_path(gedi_cfg["path"], root)
    require_file(gedi_path, "GEDI points")
    rh98_field = str(gedi_cfg.get("rh98_field", "rh98"))
    quality_field = str(gedi_cfg.get("quality_flag_field", "quality_flag"))
    degrade_field = str(gedi_cfg.get("degrade_flag_field", "degrade_flag"))
    modes_field = str(gedi_cfg.get("num_detectedmodes_field", "num_detectedmodes"))
    sensitivity_field = str(gedi_cfg.get("sensitivity_field", "sensitivity"))
    points = gpd.read_file(gedi_path)
    if points.crs is None:
        raise ValueError("GEDI points have no CRS.")
    required_fields = [rh98_field, quality_field, degrade_field, modes_field, sensitivity_field]
    missing_fields = [field for field in required_fields if field not in points.columns]
    if missing_fields:
        raise KeyError(
            f"GEDI fields are absent: {missing_fields}. Available: {list(points.columns)}"
        )
    if not np.all(points.geometry.geom_type == "Point"):
        raise ValueError("GEDI input must contain point geometries only.")

    summary = {"gedi_input": int(len(points))}
    points = points.to_crs(output_crs)[required_fields + ["geometry"]].rename(
        columns={
            rh98_field: "gedi_rh98",
            quality_field: "quality_flag",
            degrade_field: "degrade_flag",
            modes_field: "num_detectedmodes",
            sensitivity_field: "sensitivity",
        }
    )
    points = apply_gedi_quality_control(
        points,
        cfg["sample_selection"]["gedi_quality_control"],
        summary,
    )
    minimum = float(cfg["sample_selection"]["tall"]["minimum_height_m"])
    points = points[np.isfinite(points["gedi_rh98"]) & (points["gedi_rh98"] >= minimum)].copy()
    summary["gedi_after_height"] = int(len(points))

    points = attach_reference_heights(points, glo_spec, g3due_spec)
    rd_cfg = cfg["sample_selection"]["relative_difference"]
    epsilon = float(rd_cfg.get("epsilon", 1e-6))
    # H1 = GEDI and H2 = the corresponding reference-height product.
    points["rd_gedi_glo"] = relative_height_difference(
        points["gedi_rh98"], points["glo_m"], epsilon
    )
    points["rd_gedi_g3due"] = relative_height_difference(
        points["gedi_rh98"], points["g3due_m"], epsilon
    )
    threshold = float(cfg["sample_selection"]["tall"]["maximum_rd"])
    valid = (
        np.isfinite(points[["glo_m", "g3due_m", "rd_gedi_glo", "rd_gedi_g3due"]]).all(axis=1)
        & (points["glo_m"] > 0)
        & (points["g3due_m"] > 0)
        & (points["rd_gedi_glo"] < threshold)
        & (points["rd_gedi_g3due"] < threshold)
    )
    points = points[valid].copy()
    summary["gedi_after_rd_filter"] = int(len(points))
    points["height_label_m"] = points[["gedi_rh98", "glo_m", "g3due_m"]].mean(axis=1)
    points["sample_source"] = "GEDI+3D-GloBFP+G-3DUE"
    points["selection_status"] = "selected"
    return points, summary


def prepare_low_rise_samples(
    cfg: dict[str, Any], root: Path, output_crs: str, glo_spec: RasterSpec,
    g3due_spec: RasterSpec, seed: int
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    points = load_candidate_points(cfg, root, output_crs, glo_spec, seed)
    summary = {"low_rise_candidates": int(len(points))}
    points = attach_reference_heights(points, glo_spec, g3due_spec)

    low_cfg = cfg["sample_selection"]["low_rise"]
    minimum = float(low_cfg.get("minimum_height_m", 0.0))
    maximum = float(low_cfg.get("maximum_height_m", 5.0))
    rd_cfg = cfg["sample_selection"]["relative_difference"]
    epsilon = float(rd_cfg.get("epsilon", 1e-6))
    # H1 = 3D-GloBFP and H2 = G-3DUE for the independent low-rise samples.
    points["rd_glo_g3due"] = relative_height_difference(
        points["glo_m"], points["g3due_m"], epsilon
    )
    valid = (
        np.isfinite(points[["glo_m", "g3due_m", "rd_glo_g3due"]]).all(axis=1)
        & (points["glo_m"] > minimum)
        & (points["glo_m"] < maximum)
        & (points["g3due_m"] > minimum)
        & (points["g3due_m"] < maximum)
        & (points["rd_glo_g3due"] < float(low_cfg["maximum_rd"]))
    )
    points = points[valid].copy()
    summary["low_rise_after_rd_filter"] = int(len(points))
    points["gedi_rh98"] = np.nan
    points["quality_flag"] = np.nan
    points["degrade_flag"] = np.nan
    points["num_detectedmodes"] = np.nan
    points["sensitivity"] = np.nan
    points["height_label_m"] = points[["glo_m", "g3due_m"]].mean(axis=1)
    points["sample_source"] = "3D-GloBFP+G-3DUE"
    points["selection_status"] = "selected"
    return points, summary


def apply_shared_filters(
    points: gpd.GeoDataFrame,
    cfg: dict[str, Any],
    root: Path,
    icpis_core_path: Path,
    annual_planet_specs: Sequence[RasterSpec],
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    mask_cfg = cfg["sample_selection"]["impervious_core"]
    allowed_values = {float(v) for v in mask_cfg.get("impervious_values", [1])}
    summary: dict[str, int] = {"before_shared_filters": int(len(points))}

    core = core_impervious_mask(
        points,
        icpis_core_path,
        allowed_values,
        radius_m=float(mask_cfg.get("radius_m", 12.5)),
        radius_pixels=(
            int(mask_cfg["core_radius_pixels"])
            if mask_cfg.get("core_radius_pixels") is not None
            else None
        ),
    )
    points = points[core].copy()
    summary["after_impervious_core"] = int(len(points))
    # ICPIS is a single expansion-year raster: 0 is non-impervious, 2016 is
    # baseline impervious surface, and 2017-2023 are newly expanded pixels.
    # Requiring the full erosion neighborhood to equal 2016 excludes expansion.
    points["icpis_class"] = min(allowed_values) if len(allowed_values) == 1 else 2016
    points["icpis_stable"] = True

    sd_cfg = cfg["sample_selection"]["planet_sd"]
    precomputed_sd_cfg = cfg["inputs"].get("planet_gray_sd")
    if precomputed_sd_cfg and precomputed_sd_cfg.get("path"):
        # Preferred when gray_std was exported from the supplied GEE workflow.
        # Sampling this raster avoids any difference in annual compositing or
        # projection between Earth Engine and the local Python environment.
        spectral_sd = sample_raster(
            points, raster_spec(precomputed_sd_cfg, root)
        )[:, 0]
        annual_gray = None
        valid_years = None
    else:
        spectral_sd, annual_gray, valid_years = planet_spectral_sd(
            points,
            annual_planet_specs,
            rgb_weights=sd_cfg.get("rgb_weights", [0.3, 0.59, 0.11]),
            ddof=int(sd_cfg.get("ddof", 0)),
            minimum_valid_years=int(sd_cfg.get("minimum_valid_years", 2)),
        )
    points["spectral_sd"] = spectral_sd
    if annual_gray is not None:
        points["gray_valid_years"] = valid_years
        planet_years = [
            int(item.get("year", idx))
            for idx, item in enumerate(cfg["inputs"]["planet_annual"], start=1)
        ]
        for idx, year in enumerate(planet_years):
            points[f"gray_{year}"] = annual_gray[:, idx]
    minimum_sd = float(sd_cfg["minimum"])
    maximum_sd = float(sd_cfg["maximum"])
    sd_valid = np.isfinite(spectral_sd) & (spectral_sd >= minimum_sd) & (spectral_sd <= maximum_sd)
    points = points[sd_valid].copy()
    summary["after_planet_sd"] = int(len(points))
    return points, summary


def cap_low_rise_count(
    points: gpd.GeoDataFrame, target_count: int, seed: int
) -> gpd.GeoDataFrame:
    if len(points) <= target_count:
        return points
    return points.sample(n=target_count, random_state=seed).copy()


def output_paths(cfg: dict[str, Any], root: Path) -> dict[str, Path]:
    out_cfg = cfg["outputs"]
    directory = resolve_path(out_cfg.get("directory", "outputs/sample_preparation"), root)
    stem = str(out_cfg.get("stem", "training_samples"))
    return {
        "directory": directory,
        "gpkg": directory / f"{stem}.gpkg",
        "csv": directory / f"{stem}.csv",
        "summary": directory / f"{stem}_summary.json",
    }


def check_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    existing = [path for key, path in paths.items() if key != "directory" and path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Outputs already exist. Use --overwrite:\n{joined}")


def export_results(
    points: gpd.GeoDataFrame, paths: dict[str, Path], summary: dict[str, Any], output_crs: str
) -> None:
    result = points.to_crs(output_crs).reset_index(drop=True)
    result.insert(0, "sample_id", [f"S{i + 1:08d}" for i in range(len(result))])
    result["longitude"] = result.geometry.x
    result["latitude"] = result.geometry.y

    result.to_file(paths["gpkg"], layer="training_samples", driver="GPKG")
    csv_table = pd.DataFrame(result.drop(columns="geometry"))
    csv_table.to_csv(paths["csv"], index=False, encoding="utf-8")
    with paths["summary"].open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    cfg, root = load_config(args.config.resolve())
    setup_logging(str(cfg.get("logging", {}).get("level", "INFO")))
    seed = int(cfg.get("random_seed", 2023))
    output_crs = str(cfg.get("output_crs", "EPSG:4326"))

    glo_spec = raster_spec(cfg["inputs"]["glo_bfp"], root)
    g3due_spec = raster_spec(cfg["inputs"]["g3due"], root)
    icpis_core_path = resolve_path(cfg["inputs"]["icpis_core"]["path"], root)
    require_file(icpis_core_path, "ICPIS core mask")

    precomputed_sd_cfg = cfg["inputs"].get("planet_gray_sd")
    if precomputed_sd_cfg and precomputed_sd_cfg.get("path"):
        # Validate now; it will be sampled after the reference-height filters.
        raster_spec(precomputed_sd_cfg, root)
        annual_planet_specs: list[RasterSpec] = []
    else:
        annual_planet_specs = [
            raster_spec(item, root, default_bands=(1, 2, 3))
            for item in cfg["inputs"]["planet_annual"]
        ]
        if len(annual_planet_specs) < 2:
            raise ValueError("At least two annual Planet composites are required.")
        band_counts = {len(spec.bands) for spec in annual_planet_specs}
        if len(band_counts) != 1:
            raise ValueError("All annual Planet specifications must select the same number of bands.")
        if next(iter(band_counts)) != 3:
            raise ValueError(
                "Each annual Planet specification must select exactly three bands: R, G and B."
            )

    paths = output_paths(cfg, root)
    check_outputs(paths, args.overwrite)

    LOGGER.info("Preparing GEDI-based samples")
    tall, tall_summary = prepare_tall_samples(
        cfg, root, output_crs, glo_spec, g3due_spec
    )
    tall, tall_filter_summary = apply_shared_filters(
        tall, cfg, root, icpis_core_path, annual_planet_specs
    )

    LOGGER.info("Preparing independent low-rise samples")
    low, low_summary = prepare_low_rise_samples(
        cfg, root, output_crs, glo_spec, g3due_spec, seed
    )
    low, low_filter_summary = apply_shared_filters(
        low, cfg, root, icpis_core_path, annual_planet_specs
    )
    target_low = int(cfg["sample_selection"]["low_rise"]["target_count"])
    low = cap_low_rise_count(low, target_low, seed)

    combined = pd.concat([tall, low], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=output_crs)
    combined = combined.drop_duplicates(subset=["geometry", "sample_source"]).copy()

    summary: dict[str, Any] = {
        "configuration": str(args.config.resolve()),
        "random_seed": seed,
        "rd_filter_applied": True,
        "rd_formula": "abs((H1-H2)/H2)",
        "rd_pair_order": {
            "tall_1": "H1=GEDI, H2=3D-GloBFP",
            "tall_2": "H1=GEDI, H2=G-3DUE",
            "low_rise": "H1=3D-GloBFP, H2=G-3DUE",
        },
        "selection_status": "final_selected_samples",
        "tall_samples": {**tall_summary, **tall_filter_summary, "final": int(len(tall))},
        "low_rise_samples": {**low_summary, **low_filter_summary, "final": int(len(low))},
        "total_final": int(len(combined)),
    }
    LOGGER.info("Exporting %d samples", len(combined))
    export_results(combined, paths, summary, output_crs)
    LOGGER.info("Done: %s", paths["directory"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # concise CLI failure with a non-zero exit status
        LOGGER.error("%s", exc)
        raise
