"""Unduh NOAA OISST v2.1 dan susun panel SST bulanan grid penelitian.

Input:
  D:/thesis/data/metadata/output_grid_target.csv

Output utama:
  D:/thesis/data/processed/sst/sst_monthly_200301_202512_v1.csv

Sumber:
  NOAA OISST v2.1 harian, resolusi 0,25 derajat.
  https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


VERSION = "v1"
PRODUCT = "NOAA OISST v2.1 AVHRR"
DOI = "10.25921/RE9P-PT57"
BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
)
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


def dates_in_month(month: pd.Timestamp) -> pd.DatetimeIndex:
    last_day = calendar.monthrange(month.year, month.month)[1]
    return pd.date_range(month, month.replace(day=last_day), freq="D")


def source_url(day: pd.Timestamp) -> str:
    filename = f"oisst-avhrr-v02r01.{day:%Y%m%d}.nc"
    return f"{BASE_URL}/{day:%Y%m}/{filename}"


def build_session(workers: int) -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=workers,
        pool_maxsize=workers,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "IPB-thesis-OISST-downloader/1.0"})
    session.mount("https://", adapter)
    return session


def download_day(
    session: requests.Session,
    day: pd.Timestamp,
    cache_dir: Path,
    timeout: int,
) -> tuple[pd.Timestamp, Path, str]:
    url = source_url(day)
    destination = cache_dir / Path(url).name
    temporary = destination.with_suffix(destination.suffix + ".part")

    if destination.is_file() and destination.stat().st_size > 0:
        return day, destination, url

    with session.get(url, stream=True, timeout=(30, timeout)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"File kosong: {url}")
    temporary.replace(destination)
    return day, destination, url


def extract_daily_subset(path: Path, west: float, east: float,
                         south: float, north: float) -> xr.Dataset:
    with xr.open_dataset(path, engine="h5netcdf", mask_and_scale=True) as source:
        needed = {"sst", "anom", "lat", "lon", "time"}
        missing = sorted(needed - set(source.variables))
        if missing:
            raise ValueError(f"Variabel OISST tidak lengkap pada {path.name}: {missing}")

        data = source[["sst", "anom"]]
        if "zlev" in data.dims:
            data = data.isel(zlev=0, drop=True)

        lat_values = np.asarray(data.lat.values)
        lon_values = np.asarray(data.lon.values)
        lat_index = np.flatnonzero((lat_values >= south) & (lat_values <= north))
        lon_index = np.flatnonzero((lon_values >= west) & (lon_values <= east))
        if not len(lat_index) or not len(lon_index):
            raise ValueError(f"Wilayah penelitian tidak tercakup oleh {path.name}.")

        subset = data.isel(lat=lat_index, lon=lon_index).load()
        subset.attrs = {
            "source_product": PRODUCT,
            "source_filename": path.name,
            "source_doi": DOI,
        }
        return subset


def write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoding = {
        variable: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for variable in ["sst", "anom"]
    }
    dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    temporary.replace(path)


def create_month_subset(
    month: pd.Timestamp,
    subset_path: Path,
    cache_dir: Path,
    bbox: tuple[float, float, float, float],
    workers: int,
    timeout: int,
    keep_cache: bool,
) -> xr.Dataset:
    if subset_path.is_file():
        with xr.open_dataset(subset_path, engine="h5netcdf", mask_and_scale=True) as ds:
            loaded = ds.load()
        expected_days = len(dates_in_month(month))
        if len(loaded.time) != expected_days:
            raise ValueError(
                f"Subset {subset_path.name} hanya memiliki {len(loaded.time)} hari; "
                f"seharusnya {expected_days}."
            )
        return loaded

    cache_dir.mkdir(parents=True, exist_ok=True)
    days = dates_in_month(month)
    session = build_session(workers)
    downloaded: dict[pd.Timestamp, tuple[Path, str]] = {}

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_day, session, day, cache_dir, timeout): day
                for day in days
            }
            for future in as_completed(futures):
                day, path, url = future.result()
                downloaded[day] = (path, url)

        west, south, east, north = bbox
        daily = []
        urls = []
        for day in days:
            path, url = downloaded[day]
            subset = extract_daily_subset(path, west, east, south, north)
            daily.append(subset)
            urls.append(url)

        combined = xr.concat(daily, dim="time").sortby("time")
        combined.attrs = {
            "title": "Geographic subset of NOAA OISST v2.1 daily data",
            "source_product": PRODUCT,
            "source_doi": DOI,
            "source_base_url": BASE_URL,
            "month": month.strftime("%Y-%m"),
            "downloaded_files": len(urls),
            "processing_note": "Spatial subset only; daily values are unchanged.",
        }
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        write_netcdf(combined, subset_path)
        return combined
    finally:
        session.close()
        if not keep_cache:
            for path, _ in downloaded.values():
                path.unlink(missing_ok=True)


def nearest_indices(source: np.ndarray, target: np.ndarray,
                    tolerance: float = 0.126) -> np.ndarray:
    distance = np.abs(source[:, None] - target[None, :])
    indices = distance.argmin(axis=0)
    nearest_distance = distance[indices, np.arange(len(target))]
    if (nearest_distance > tolerance).any():
        bad = np.flatnonzero(nearest_distance > tolerance)[0]
        raise ValueError(
            f"Tidak ada piksel OISST dekat koordinat target {target[bad]:.6f}."
        )
    return indices


def aggregate_month(grid: pd.DataFrame, month: pd.Timestamp,
                    daily: xr.Dataset) -> list[dict]:
    lat = np.asarray(daily.lat.values, dtype=float)
    lon = np.asarray(daily.lon.values, dtype=float)
    lat_index = nearest_indices(lat, grid.lat.to_numpy(dtype=float))
    lon_index = nearest_indices(lon, grid.lon.to_numpy(dtype=float))
    n_days_total = int(daily.sizes["time"])
    records = []

    for position, cell in enumerate(grid.itertuples(index=False)):
        sst = np.asarray(
            daily.sst.isel(lat=lat_index[position], lon=lon_index[position]).values,
            dtype=float,
        ).reshape(-1)
        anomaly = np.asarray(
            daily.anom.isel(lat=lat_index[position], lon=lon_index[position]).values,
            dtype=float,
        ).reshape(-1)
        valid_sst = np.isfinite(sst)
        valid_anomaly = np.isfinite(anomaly)
        n_valid_days = int(valid_sst.sum())

        records.append({
            "month": month,
            "grid_id": cell.grid_id,
            "lon": cell.lon,
            "lat": cell.lat,
            "oisst_lon": float(lon[lon_index[position]]),
            "oisst_lat": float(lat[lat_index[position]]),
            "sst_c": float(np.mean(sst[valid_sst])) if n_valid_days else np.nan,
            "sst_anomaly_c": (
                float(np.mean(anomaly[valid_anomaly]))
                if valid_anomaly.any() else np.nan
            ),
            "n_valid_days": n_valid_days,
            "n_days_total": n_days_total,
            "valid_day_fraction": n_valid_days / n_days_total,
            "source_status": "ok",
        })
    return records


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\thesis"))
    parser.add_argument("--grid", type=Path,
                        help="Default: data/metadata/output_grid_target.csv")
    parser.add_argument("--start", type=parse_month, default=pd.Timestamp("2003-01-01"))
    parser.add_argument("--end", type=parse_month, default=pd.Timestamp("2025-12-01"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180,
                        help="Batas waktu baca tiap unduhan dalam detik.")
    parser.add_argument("--check-grid", action="store_true")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Simpan file global harian setelah subset dibuat.")
    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error("--start tidak boleh melebihi --end.")
    if not 1 <= args.workers <= 8:
        parser.error("--workers harus berada pada rentang 1 sampai 8.")

    root = args.root.expanduser()
    grid_path = args.grid or root / "data" / "metadata" / "output_grid_target.csv"
    grid = load_grid(grid_path)
    months = month_range(args.start, args.end)
    bbox = (
        float(grid.west.min()) - 0.125,
        float(grid.south.min()) - 0.125,
        float(grid.east.max()) + 0.125,
        float(grid.north.max()) + 0.125,
    )

    print(
        f"Grid: {len(grid)} | Bulan: {len(months)} | "
        f"Periode: {args.start:%Y-%m} s.d. {args.end:%Y-%m}"
    )
    if args.check_grid:
        return 0

    raw_root = root / "data" / "raw" / "sst" / "OISST_v2.1_daily_subset"
    cache_dir = root / "data" / "raw" / "sst" / ".cache_daily_global"
    processed = root / "data" / "processed" / "sst"
    processed.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start:%Y%m}_{args.end:%Y%m}_{VERSION}"
    status_path = processed / f"download_status_sst_monthly_{tag}.csv"

    records = []
    statuses = []
    for number, month in enumerate(months, start=1):
        subset_path = (
            raw_root / f"{month:%Y}" / f"oisst_daily_subset_{month:%Y%m}_{VERSION}.nc"
        )
        status = {
            "month": month,
            "raw_subset": str(subset_path),
            "status": "",
        }
        try:
            monthly_daily = create_month_subset(
                month=month,
                subset_path=subset_path,
                cache_dir=cache_dir,
                bbox=bbox,
                workers=args.workers,
                timeout=args.timeout,
                keep_cache=args.keep_cache,
            )
            records.extend(aggregate_month(grid, month, monthly_daily))
            status["status"] = "ok"
            status["n_daily_files"] = int(monthly_daily.sizes["time"])
        except Exception as exc:
            status.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            statuses.append(status)
            atomic_csv(pd.DataFrame(statuses), status_path)
            raise RuntimeError(
                f"Berhenti pada {month:%Y-%m}: {type(exc).__name__}: {exc}"
            ) from exc

        statuses.append(status)
        atomic_csv(pd.DataFrame(statuses), status_path)
        print(f"[{number}/{len(months)}] {month:%Y-%m}: ok", flush=True)

    monthly = pd.DataFrame(records)
    expected_rows = len(grid) * len(months)
    if len(monthly) != expected_rows:
        raise RuntimeError(
            f"Jumlah baris {len(monthly)} tidak sama dengan target {expected_rows}."
        )

    monthly_path = processed / f"sst_monthly_{tag}.csv"
    atomic_csv(monthly, monthly_path)

    metadata = {
        "script_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "doi": DOI,
        "source_base_url": BASE_URL,
        "start": args.start.strftime("%Y-%m"),
        "end": args.end.strftime("%Y-%m"),
        "periods_requested": len(months),
        "grid_count": len(grid),
        "expected_rows": expected_rows,
        "grid_file": str(grid_path),
        "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "source_resolution": "0.25 degree daily",
        "target_resolution": "0.25 degree monthly",
        "temporal_method": "Arithmetic mean of valid daily OISST values per month",
        "spatial_method": "Nearest OISST grid-cell center within 0.126 degree",
    }
    metadata_path = processed / f"run_metadata_sst_monthly_{tag}.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(metadata_path)

    print(f"Selesai: {len(monthly)} baris bulanan.")
    print(f"CSV: {monthly_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDihentikan. Subset bulanan yang sudah selesai tetap tersimpan.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
