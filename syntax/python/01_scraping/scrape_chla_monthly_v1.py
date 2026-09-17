"""Unduh MODIS Aqua Chl-a Level-3 bulanan dan agregasikan ke grid 0,25°.

Produk sumber: MODISA_L3m_CHL (R2022.0), komposit bulanan 4 km.
Default: Januari 2003 sampai Desember 2025.

Input:
  D:/thesis/data/metadata/output_grid_target.csv
Output:
  D:/thesis/data/raw/chla/MODISA_2022.0_4km_monthly/<tahun>/
  D:/thesis/data/processed/chla/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import xarray as xr


VERSION = "v1"
COLLECTION = "C3380709133-OB_CLOUD"
PRODUCT = "MODISA_L3m_CHL"
PRODUCT_VERSION = "2022.0"
PATTERN = "AQUA_MODIS.*.L3m.MO.CHL.chlor_a.4km.nc"
FILENAME_RE = re.compile(
    r"^AQUA_MODIS\.(\d{8})_(\d{8})\.L3m\.MO\.CHL\.chlor_a\.4km\.nc$"
)
COORDS = ["lon", "lat", "west", "east", "south", "north"]


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
        raise ValueError(f"CSV grid tidak ditemukan: {path}")
    grid = pd.read_csv(path, dtype={"grid_id": str})
    required = ["grid_id", *COORDS]
    missing = sorted(set(required) - set(grid.columns))
    if missing:
        raise ValueError(f"Kolom grid belum lengkap: {missing}")
    if "selected" in grid:
        selected = grid["selected"].astype(str).str.strip().str.lower()
        if not selected.isin(["true", "false", "1", "0"]).all():
            raise ValueError("Kolom selected harus TRUE/FALSE atau 1/0.")
        grid = grid.loc[selected.isin(["true", "1"])].copy()
    if grid.empty or grid.grid_id.isna().any() or grid.grid_id.duplicated().any():
        raise ValueError("Grid kosong atau grid_id kosong/duplikat.")
    for column in COORDS:
        grid[column] = pd.to_numeric(grid[column], errors="raise")
    if not np.isfinite(grid[COORDS].to_numpy()).all():
        raise ValueError("Koordinat grid mengandung nilai kosong atau tidak hingga.")
    if not (np.allclose(grid.east - grid.west, 0.25)
            and np.allclose(grid.north - grid.south, 0.25)):
        raise ValueError("Ukuran grid harus 0,25 x 0,25 derajat.")
    if not (np.allclose(grid.lon, (grid.west + grid.east) / 2)
            and np.allclose(grid.lat, (grid.south + grid.north) / 2)):
        raise ValueError("Koordinat pusat tidak sesuai batas grid.")
    return grid.reset_index(drop=True)


def bbox_of(grid: pd.DataFrame) -> tuple[float, float, float, float]:
    return (float(grid.west.min()), float(grid.south.min()),
            float(grid.east.max()), float(grid.north.max()))


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="MS")


def select_granules(results, start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, dict]:
    """Pilih satu komposit resmi NASA per bulan."""
    selected: dict[pd.Timestamp, dict] = {}
    for granule in results:
        matches = []
        for url in granule.data_links(access="external"):
            if not url.startswith("https://"):
                continue
            filename = unquote(Path(urlparse(url).path).name)
            match = FILENAME_RE.fullmatch(filename)
            if match:
                matches.append((url, match))
        if not matches:
            continue
        url, match = matches[0]
        start_day = pd.Timestamp(match.group(1))
        end_day = pd.Timestamp(match.group(2))
        month = start_day.to_period("M").to_timestamp()
        if not (start <= month <= end):
            continue
        if end_day.to_period("M") != month.to_period("M"):
            raise ValueError(f"Komposit bulanan lintas-bulan tidak diharapkan: {match.group(0)}")
        metadata = granule.get("meta", {})
        entry = {
            "granule": granule,
            "filename": match.group(0),
            "source_url": url,
            "concept_id": metadata.get("concept-id", ""),
            "revision_date": metadata.get("revision-date", ""),
        }
        old = selected.get(month)
        if old is None or entry["revision_date"] > old["revision_date"]:
            selected[month] = entry
    return selected


def discover(earthaccess, start: pd.Timestamp, end: pd.Timestamp, bbox) -> dict:
    results = earthaccess.search_data(
        concept_id=COLLECTION,
        temporal=(f"{start:%Y-%m-%d}T00:00:00Z", f"{end:%Y-%m-%d}T23:59:59Z"),
        bounding_box=bbox,
        granule_name=PATTERN,
        count=-1,
    )
    return select_granules(results, start, end)


def read_subset(path: Path, bbox):
    west, south, east, north = bbox
    with xr.open_dataset(path, engine="h5netcdf", mask_and_scale=True) as ds:
        if not {"chlor_a", "lat", "lon"}.issubset(ds.variables):
            raise ValueError(f"Variabel chlor_a/lat/lon tidak ditemukan: {path.name}")
        if ds.lat.ndim != 1 or ds.lon.ndim != 1:
            raise ValueError("Produk bukan raster Level-3 mapped dengan koordinat satu dimensi.")
        lat, lon = np.asarray(ds.lat.values), np.asarray(ds.lon.values)
        for values in (lat, lon):
            delta = np.diff(values)
            if not (np.isfinite(values).all() and (np.all(delta > 0) or np.all(delta < 0))):
                raise ValueError("Koordinat produk tidak terurut atau tidak valid.")
            if not np.allclose(np.abs(delta), 1 / 24, atol=2e-5, rtol=0):
                raise ValueError("Resolusi produk bukan L3 mapped nominal 4 km.")
        iy = np.flatnonzero((lat >= south) & (lat < north))
        ix = np.flatnonzero((lon >= west) & (lon < east))
        if not len(iy) or not len(ix):
            raise ValueError("File tidak mencakup wilayah studi.")
        da = ds.chlor_a.isel(lat=iy, lon=ix).transpose("lat", "lon").load()
        values = np.asarray(da.values, dtype=float)
        valid = np.isfinite(values) & (values > 0)
        if "valid_min" in da.attrs:
            valid &= values >= float(da.attrs["valid_min"])
        if "valid_max" in da.attrs:
            valid &= values <= float(da.attrs["valid_max"])
        values[~valid] = np.nan
        metadata = {key: str(ds.attrs[key]) for key in
                    ["processing_version", "date_created", "time_coverage_start", "time_coverage_end"]
                    if key in ds.attrs}
    return values, lat[iy], lon[ix], metadata


def aggregate_month(grid: pd.DataFrame, month: pd.Timestamp, subset=None,
                    source_status: str = "ok") -> list[dict]:
    records = []
    for cell in grid.itertuples(index=False):
        mean, n_valid, n_total = np.nan, 0, None
        if subset is not None:
            values, lat, lon, _ = subset
            iy = np.flatnonzero((lat >= cell.south) & (lat < cell.north))
            ix = np.flatnonzero((lon >= cell.west) & (lon < cell.east))
            if not len(iy) or not len(ix):
                raise ValueError(f"Tidak ada pusat piksel pada {cell.grid_id}.")
            pixels = values[np.ix_(iy, ix)]
            weights = np.broadcast_to(np.cos(np.deg2rad(lat[iy]))[:, None], pixels.shape)
            valid = np.isfinite(pixels)
            n_total, n_valid = pixels.size, int(valid.sum())
            if n_valid:
                mean = float(np.average(pixels[valid], weights=weights[valid]))
        records.append({
            "month": month, "grid_id": cell.grid_id, "lon": cell.lon, "lat": cell.lat,
            "chla_mg_m3": mean, "n_valid_pixels": n_valid,
            "n_pixels_total": n_total,
            "valid_pixel_fraction": (n_valid / n_total) if n_total else np.nan,
            "source_status": source_status,
        })
    return records


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", na_rep="")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\thesis"))
    parser.add_argument("--grid", type=Path, help="Default: data/metadata/output_grid_target.csv")
    parser.add_argument("--start", type=parse_month, default=pd.Timestamp("2003-01-01"))
    parser.add_argument("--end", type=parse_month, default=pd.Timestamp("2025-12-01"))
    parser.add_argument("--check-grid", action="store_true", help="Cek CSV grid tanpa jaringan")
    parser.add_argument("--catalog-only", action="store_true", help="Cek jumlah komposit NASA tanpa unduh")
    parser.add_argument("--offline", action="store_true", help="Olah NetCDF bulanan yang sudah ada")
    args = parser.parse_args(argv)
    if args.offline and args.catalog_only:
        parser.error("--offline dan --catalog-only tidak dapat digabung.")
    if args.start > args.end:
        parser.error("--start tidak boleh melebihi --end.")

    root = args.root.expanduser()
    grid_path = args.grid or root / "data" / "metadata" / "output_grid_target.csv"
    grid = load_grid(grid_path)
    bbox = bbox_of(grid)
    months = month_range(args.start, args.end)
    print(f"Grid: {len(grid)} | Bulan: {len(months)} | Batas: {bbox[0]}..{bbox[2]} BT, {bbox[1]}..{bbox[3]} LS")
    if args.check_grid:
        return 0

    raw_root = root / "data" / "raw" / "chla" / "MODISA_2022.0_4km_monthly"
    processed = root / "data" / "processed" / "chla"
    processed.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start:%Y%m}_{args.end:%Y%m}_{VERSION}"
    status_path = processed / f"download_status_monthly_{tag}.csv"

    discovered, earthaccess = {}, None
    if not args.offline:
        import earthaccess as ea
        earthaccess = ea
        print("Mencari komposit bulanan di katalog NASA...")
        discovered = discover(earthaccess, args.start, args.end, bbox)
        print(f"Komposit ditemukan: {len(discovered)}/{len(months)}")
        if args.catalog_only:
            return 0
        print("Login Earthdata di terminal. Password tidak disimpan script.")
        auth = earthaccess.login(strategy="interactive", persist=False)
        if not auth.authenticated:
            raise RuntimeError("Login Earthdata gagal.")

    records, statuses = [], []
    for number, month in enumerate(months, 1):
        entry = discovered.get(month)
        filename = entry["filename"] if entry else f"AQUA_MODIS.{month:%Y%m}01_YYYYMMDD.L3m.MO.CHL.chlor_a.4km.nc"
        raw_dir = raw_root / f"{month:%Y}"
        path = raw_dir / filename
        status = {
            "month": month, "filename": filename,
            "source_url": entry["source_url"] if entry else "",
            "concept_id": entry["concept_id"] if entry else "",
            "local_file": str(path), "status": "",
        }
        try:
            if not args.offline and entry is None:
                status["status"] = "not_in_catalog"
                records.extend(aggregate_month(grid, month, source_status="not_in_catalog"))
            else:
                raw_dir.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    if args.offline:
                        raise FileNotFoundError(f"File offline belum ada: {path}")
                    earthaccess.download([entry["granule"]], local_path=raw_dir, threads=1, show_progress=False)
                if not path.is_file():
                    raise RuntimeError("Unduhan tidak menghasilkan file yang diharapkan.")
                subset = read_subset(path, bbox)
                records.extend(aggregate_month(grid, month, subset))
                status.update(subset[3])
                status["status"] = "ok"
        except Exception as exc:
            status.update({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})
            statuses.append(status)
            atomic_csv(pd.DataFrame(statuses), status_path)
            raise RuntimeError(f"Berhenti pada {month:%Y-%m}: {type(exc).__name__}: {exc}") from exc
        statuses.append(status)
        atomic_csv(pd.DataFrame(statuses), status_path)
        print(f"[{number}/{len(months)}] {month:%Y-%m}: {status['status']}", flush=True)

    monthly = pd.DataFrame(records)
    monthly["n_pixels_total"] = monthly.n_pixels_total.astype("Int64")
    monthly_path = processed / f"chla_monthly_{tag}.csv"
    atomic_csv(monthly, monthly_path)
    metadata = {
        "script_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION,
        "product": PRODUCT,
        "product_version": PRODUCT_VERSION,
        "doi": "10.5067/AQUA/MODIS/L3M/CHL/2022.0",
        "start": args.start.strftime("%Y-%m"),
        "end": args.end.strftime("%Y-%m"),
        "periods_requested": len(months),
        "grid_count": len(grid),
        "expected_rows": len(months) * len(grid),
        "grid_file": str(grid_path),
        "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "source_resolution": "1/24 degree, nominal 4 km",
        "target_resolution_degree": 0.25,
        "spatial_method": "cos(latitude)-weighted mean of valid native pixels in each grid",
        "temporal_method": "official NASA monthly Level-3 composite; no user aggregation of daily files",
        "missing": "NaN retained when a monthly composite has no valid pixel in a grid",
    }
    meta_path = processed / f"run_metadata_monthly_{tag}.json"
    temporary = meta_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(meta_path)
    print(f"Selesai: {len(monthly)} baris bulanan.")
    print(f"CSV: {monthly_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDihentikan. File yang sudah lengkap tetap tersimpan.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
