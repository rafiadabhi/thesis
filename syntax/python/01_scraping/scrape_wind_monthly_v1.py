"""Unduh ERA5 u10 dan v10 bulanan lalu petakan ke grid penelitian.

Prasyarat:
  1. Paket cdsapi terpasang pada virtual environment.
  2. C:\\Users\\<user>\\.cdsapirc berisi Personal Access Token CDS.
  3. Terms and Conditions dataset ERA5 sudah disetujui pada CDS.

Input:
  D:/thesis/data/metadata/output_grid_target.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


VERSION = "v1"
DATASET = "reanalysis-era5-single-levels-monthly-means"
DOI = "10.24381/cds.f17050d7"
GRID_COLUMNS = ["lon", "lat", "west", "east", "south", "north"]


def parse_month(value: str) -> pd.Timestamp:
    try:
        month = pd.Timestamp(f"{value}-01")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Gunakan format YYYY-MM.") from exc
    if month.strftime("%Y-%m") != value:
        raise argparse.ArgumentTypeError("Gunakan format YYYY-MM.")
    return month


def load_grid(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV grid tidak ditemukan: {path}")
    grid = pd.read_csv(path, dtype={"grid_id": str})
    required = ["grid_id", *GRID_COLUMNS]
    missing = sorted(set(required) - set(grid.columns))
    if missing:
        raise ValueError(f"Kolom grid belum lengkap: {missing}")

    if "selected" in grid.columns:
        selected = grid.selected.astype(str).str.strip().str.lower()
        if not selected.isin(["true", "false", "1", "0"]).all():
            raise ValueError("Kolom selected harus TRUE/FALSE atau 1/0.")
        grid = grid.loc[selected.isin(["true", "1"])].copy()
    if grid.empty or grid.grid_id.isna().any() or grid.grid_id.duplicated().any():
        raise ValueError("Grid kosong atau grid_id kosong/duplikat.")

    for column in GRID_COLUMNS:
        grid[column] = pd.to_numeric(grid[column], errors="raise")
    if not np.isfinite(grid[GRID_COLUMNS].to_numpy()).all():
        raise ValueError("Koordinat grid mengandung nilai kosong atau tidak hingga.")
    if not (
        np.allclose(grid.east - grid.west, 0.25)
        and np.allclose(grid.north - grid.south, 0.25)
    ):
        raise ValueError("Ukuran grid harus 0,25 x 0,25 derajat.")
    if not (
        np.allclose(grid.lon, (grid.west + grid.east) / 2)
        and np.allclose(grid.lat, (grid.south + grid.north) / 2)
    ):
        raise ValueError("Koordinat pusat tidak sesuai batas grid.")
    return grid.sort_values("grid_id").reset_index(drop=True)


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="MS")


def find_name(dataset: xr.Dataset, candidates: list[str], kind: str) -> str:
    for candidate in candidates:
        if candidate in dataset.variables:
            return candidate
    raise ValueError(
        f"Variabel {kind} tidak ditemukan. Tersedia: {sorted(dataset.variables)}"
    )


def nearest_indices(source: np.ndarray, target: np.ndarray,
                    tolerance: float = 0.126) -> np.ndarray:
    distance = np.abs(source[:, None] - target[None, :])
    indices = distance.argmin(axis=0)
    nearest_distance = distance[indices, np.arange(len(target))]
    if (nearest_distance > tolerance).any():
        bad = np.flatnonzero(nearest_distance > tolerance)[0]
        raise ValueError(
            f"Tidak ada piksel ERA5 dekat koordinat target {target[bad]:.6f}."
        )
    return indices


def normalize_months(time_values: np.ndarray) -> pd.DatetimeIndex:
    months = pd.DatetimeIndex(pd.to_datetime(time_values)).to_period("M").to_timestamp()
    return pd.DatetimeIndex(months)


def extract_panel(grid: pd.DataFrame, raw_path: Path,
                  requested_months: pd.DatetimeIndex) -> pd.DataFrame:
    with xr.open_dataset(raw_path, engine="h5netcdf", mask_and_scale=True) as ds:
        lon_name = find_name(ds, ["longitude", "lon"], "longitude")
        lat_name = find_name(ds, ["latitude", "lat"], "latitude")
        time_name = find_name(ds, ["valid_time", "time"], "waktu")
        u_name = find_name(ds, ["u10", "10u", "10m_u_component_of_wind"], "u10")
        v_name = find_name(ds, ["v10", "10v", "10m_v_component_of_wind"], "v10")

        longitude = np.asarray(ds[lon_name].values, dtype=float)
        latitude = np.asarray(ds[lat_name].values, dtype=float)
        months = normalize_months(np.asarray(ds[time_name].values))
        if months.duplicated().any():
            raise ValueError("File ERA5 berisi lebih dari satu nilai untuk bulan yang sama.")
        if not months.equals(requested_months):
            raise ValueError(
                f"Bulan ERA5 tidak sesuai permintaan: {months.min():%Y-%m} s.d. "
                f"{months.max():%Y-%m}, jumlah {len(months)}."
            )

        u = ds[u_name]
        v = ds[v_name]
        rename_dims = {}
        if time_name != "time":
            rename_dims[time_name] = "time"
        if lat_name != "lat":
            rename_dims[lat_name] = "lat"
        if lon_name != "lon":
            rename_dims[lon_name] = "lon"
        if rename_dims:
            u = u.rename(rename_dims)
            v = v.rename(rename_dims)
        u = u.transpose("time", "lat", "lon").load()
        v = v.transpose("time", "lat", "lon").load()

    lat_index = nearest_indices(latitude, grid.lat.to_numpy(dtype=float))
    lon_index = nearest_indices(longitude, grid.lon.to_numpy(dtype=float))
    records = []
    for time_index, month in enumerate(requested_months):
        for position, cell in enumerate(grid.itertuples(index=False)):
            u_value = float(u.values[time_index, lat_index[position], lon_index[position]])
            v_value = float(v.values[time_index, lat_index[position], lon_index[position]])
            speed = np.sqrt(u_value ** 2 + v_value ** 2)
            direction_from = (np.degrees(np.arctan2(-u_value, -v_value)) + 360) % 360
            records.append({
                "month": month,
                "grid_id": cell.grid_id,
                "lon": cell.lon,
                "lat": cell.lat,
                "era5_lon": float(longitude[lon_index[position]]),
                "era5_lat": float(latitude[lat_index[position]]),
                "u10_m_s": u_value,
                "v10_m_s": v_value,
                "wind_speed_m_s": float(speed),
                "wind_direction_from_deg": float(direction_from),
                "source_status": "ok",
            })
    return pd.DataFrame(records)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        na_rep="",
    )
    temporary.replace(path)


def request_era5(raw_path: Path, months: pd.DatetimeIndex,
                 area: list[float]) -> None:
    import cdsapi

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": sorted({f"{month:%Y}" for month in months}),
        "month": sorted({f"{month:%m}" for month in months}),
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(raw_path.suffix + ".part")
    client = cdsapi.Client()
    result = client.retrieve(DATASET, request)
    result.download(str(temporary))
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("CDS tidak menghasilkan NetCDF yang valid.")
    temporary.replace(raw_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\thesis"))
    parser.add_argument("--grid", type=Path,
                        help="Default: data/metadata/output_grid_target.csv")
    parser.add_argument("--start", type=parse_month, default=pd.Timestamp("2003-01-01"))
    parser.add_argument("--end", type=parse_month, default=pd.Timestamp("2025-12-01"))
    parser.add_argument("--check-grid", action="store_true")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start tidak boleh melebihi --end.")

    root = args.root.expanduser()
    grid_path = args.grid or root / "data" / "metadata" / "output_grid_target.csv"
    grid = load_grid(grid_path)
    months = month_range(args.start, args.end)
    print(
        f"Grid: {len(grid)} | Bulan: {len(months)} | "
        f"Periode: {args.start:%Y-%m} s.d. {args.end:%Y-%m}"
    )
    if args.check_grid:
        return 0

    raw_root = root / "data" / "raw" / "wind"
    processed = root / "data" / "processed" / "wind"
    processed.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start:%Y%m}_{args.end:%Y%m}_{VERSION}"
    raw_path = raw_root / f"era5_u10_v10_monthly_{tag}.nc"

    # CDS uses [north, west, south, east]. The padding preserves edge pixels.
    area = [
        float(grid.north.max()) + 0.25,
        float(grid.west.min()) - 0.25,
        float(grid.south.min()) - 0.25,
        float(grid.east.max()) + 0.25,
    ]
    if raw_path.is_file():
        print(f"Memakai file ERA5 yang sudah ada: {raw_path.name}")
    else:
        print("Mengirim permintaan ERA5 ke CDS. Waktu tunggu bergantung antrean CDS...")
        request_era5(raw_path, months, area)

    panel = extract_panel(grid, raw_path, months)
    expected_rows = len(grid) * len(months)
    if len(panel) != expected_rows:
        raise RuntimeError(
            f"Jumlah baris {len(panel)} tidak sama dengan target {expected_rows}."
        )
    output_path = processed / f"wind_monthly_{tag}.csv"
    atomic_csv(panel, output_path)

    metadata = {
        "script_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "doi": DOI,
        "start": args.start.strftime("%Y-%m"),
        "end": args.end.strftime("%Y-%m"),
        "periods_requested": len(months),
        "grid_count": len(grid),
        "expected_rows": expected_rows,
        "grid_file": str(grid_path),
        "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "source_resolution": "0.25 degree monthly mean",
        "source_variables": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "spatial_method": "Nearest ERA5 grid-cell center within 0.126 degree",
        "requested_area_nwse": area,
    }
    metadata_path = processed / f"run_metadata_wind_monthly_{tag}.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(metadata_path)
    print(f"Selesai: {len(panel)} baris bulanan.")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDihentikan. Jalankan ulang dengan perintah yang sama.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
