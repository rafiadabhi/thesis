"""Unduh MODIS Aqua Chl-a harian, agregasikan ke grid 0.25°, lalu bentuk data mingguan.

Default hanya memeriksa tiga hari pertama 2024. Sumber data:
https://doi.org/10.5067/AQUA/MODIS/L3M/CHL/2022.0

Struktur proyek yang dipakai:
  D:/thesis/Data/metadata/output_grid_target.csv
  D:/thesis/Data/raw/chla/
  D:/thesis/Data/processed/chla/
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


VERSION = "v4"
COLLECTION = "C3380709133-OB_CLOUD"
PRODUCT = "MODISA_L3m_CHL"
PRODUCT_VERSION = "2022.0"
PATTERN = "AQUA_MODIS.*.L3m.DAY.CHL.chlor_a.4km.nc"
FILENAME_RE = re.compile(r"^AQUA_MODIS\.(\d{8})\.L3m\.DAY\.CHL\.chlor_a\.4km\.nc$")
COORDS = ["lon", "lat", "west", "east", "south", "north"]


def load_grid(path: Path) -> pd.DataFrame:
    """Baca koordinat pusat dan batas setiap grid dari keluaran Rmd."""
    if not path.is_file():
        raise ValueError(f"CSV grid tidak ditemukan: {path}")
    grid = pd.read_csv(path, dtype={"grid_id": str})
    required = ["grid_id", *COORDS]
    missing = sorted(set(required) - set(grid.columns))
    if missing:
        raise ValueError(f"Kolom grid belum lengkap: {missing}")
    if "selected" in grid:
        flag = grid["selected"].astype(str).str.lower().str.strip()
        if not flag.isin(["true", "false", "1", "0"]).all():
            raise ValueError("Kolom selected harus TRUE/FALSE atau 1/0.")
        grid = grid.loc[flag.isin(["true", "1"])].copy()
    if grid.empty or grid.grid_id.isna().any() or grid.grid_id.duplicated().any():
        raise ValueError("Grid kosong atau grid_id kosong/duplikat.")
    for column in COORDS:
        grid[column] = pd.to_numeric(grid[column], errors="raise")
    if not np.isfinite(grid[COORDS].to_numpy()).all():
        raise ValueError("Koordinat grid mengandung nilai kosong atau tidak hingga.")
    if not (np.allclose(grid.east - grid.west, 0.25)
            and np.allclose(grid.north - grid.south, 0.25)):
        raise ValueError("Ukuran setiap grid harus 0.25 x 0.25 derajat.")
    if not (np.allclose(grid.lon, (grid.west + grid.east) / 2)
            and np.allclose(grid.lat, (grid.south + grid.north) / 2)):
        raise ValueError("Koordinat pusat tidak sesuai dengan batas grid.")
    if grid.duplicated(["west", "east", "south", "north"]).any():
        raise ValueError("Ada sel dengan batas yang sama.")
    return grid.reset_index(drop=True)


def bbox_of(grid: pd.DataFrame) -> tuple[float, float, float, float]:
    return (float(grid.west.min()), float(grid.south.min()),
            float(grid.east.max()), float(grid.north.max()))


def select_granules(results, start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, dict]:
    """Pilih tepat satu granule yang namanya sesuai untuk setiap tanggal."""
    selected = {}
    for granule in results:
        urls = granule.data_links(access="external")
        matches = []
        for url in urls:
            if not url.startswith("https://"):
                continue
            filename = unquote(Path(urlparse(url).path).name)
            match = FILENAME_RE.fullmatch(filename)
            if match:
                matches.append((url, match))
        if not matches:
            continue
        url, match = matches[0]
        day = pd.Timestamp(match.group(1))
        if not start <= day <= end:
            continue
        meta = granule.get("meta", {})
        entry = {
            "granule": granule,
            "filename": match.group(0),
            "source_url": url,
            "concept_id": meta.get("concept-id", ""),
            "revision_date": meta.get("revision-date", ""),
        }
        previous = selected.get(day)
        if previous is None or entry["revision_date"] > previous["revision_date"]:
            selected[day] = entry
    return selected


def discover(earthaccess, start: pd.Timestamp, end: pd.Timestamp, bbox) -> dict:
    results = earthaccess.search_data(
        concept_id=COLLECTION,
        temporal=(f"{start:%Y-%m-%d}T00:00:00Z", f"{end:%Y-%m-%d}T23:59:59Z"),
        bounding_box=bbox,
        granule_name=PATTERN,
        count=-1,
    )
    selected = select_granules(results, start, end)
    if not selected:
        raise RuntimeError("Tidak ada file harian 4 km yang cocok di katalog NASA.")
    return selected


def read_subset(path: Path, bbox):
    """Baca hanya bagian Selat Sunda. Tanggal file diverifikasi dari nama file, bukan waktu lintasan."""
    west, south, east, north = bbox
    with xr.open_dataset(path, engine="h5netcdf", mask_and_scale=True) as ds:
        if not {"chlor_a", "lat", "lon"}.issubset(ds.variables):
            raise ValueError(f"Variabel chlor_a/lat/lon tidak ditemukan: {path.name}")
        lat, lon = np.asarray(ds.lat.values), np.asarray(ds.lon.values)
        if ds.lat.ndim != 1 or ds.lon.ndim != 1:
            raise ValueError("Produk bukan raster L3 mapped dengan koordinat satu dimensi.")
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
        units = str(da.attrs.get("units", "")).replace(" ", "").lower()
        if units not in {"mgm^-3", "mgm-3", "mg/m^3", "mg/m3", "mgm**-3"}:
            raise ValueError(f"Satuan chlor_a tidak dikenali: {da.attrs.get('units')}")
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


def aggregate_day(grid: pd.DataFrame, day: pd.Timestamp, subset=None, source_status="ok") -> list[dict]:
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
            "date": day, "grid_id": cell.grid_id, "lon": cell.lon, "lat": cell.lat,
            "chla_mg_m3": mean, "n_valid_pixels": n_valid,
            "n_pixels_total": n_total, "source_status": source_status,
        })
    return records


def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["week_start"] = work.date - pd.to_timedelta(work.date.dt.dayofweek, unit="D")
    work["source_available"] = work.source_status.eq("ok").astype(int)
    weekly = work.groupby(["week_start", "grid_id"], as_index=False).agg(
        lon=("lon", "first"), lat=("lat", "first"),
        chla_mg_m3=("chla_mg_m3", "mean"),
        n_valid_days=("chla_mg_m3", "count"),
        n_days_requested=("date", "nunique"),
        n_source_days=("source_available", "sum"),
        n_valid_pixel_days=("n_valid_pixels", "sum"),
    )
    weekly.insert(1, "week_end", weekly.week_start + pd.Timedelta(days=6))
    weekly["is_partial_week"] = weekly.n_days_requested.ne(7)
    weekly["n_source_missing_days"] = weekly.n_days_requested - weekly.n_source_days
    return weekly.sort_values(["week_start", "grid_id"]).reset_index(drop=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", na_rep="")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\thesis"))
    parser.add_argument("--grid", type=Path, help="Opsional; default Data/metadata/output_grid_target.csv")
    parser.add_argument("--start", default="2024-01-01", help="Tanggal awal YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-03", help="Tanggal akhir YYYY-MM-DD")
    parser.add_argument("--full-year", action="store_true", help="Ambil satu tahun penuh berdasarkan tahun --start")
    parser.add_argument("--check-grid", action="store_true", help="Cek CSV grid saja")
    parser.add_argument("--catalog-only", action="store_true", help="Cek katalog NASA tanpa login/unduh")
    parser.add_argument("--offline", action="store_true", help="Olah NetCDF yang sudah terunduh tanpa jaringan")
    args = parser.parse_args(argv)
    if args.offline and args.catalog_only:
        parser.error("--offline dan --catalog-only tidak dapat digabung.")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if args.full_year:
        if args.end != "2024-01-03":
            parser.error("Jangan gabungkan --full-year dengan --end.")
        start = pd.Timestamp(year=start.year, month=1, day=1)
        end = pd.Timestamp(year=start.year, month=12, day=31)
    if start != start.normalize() or end != end.normalize() or start > end:
        parser.error("Gunakan tanggal YYYY-MM-DD dengan start tidak melebihi end.")

    root = args.root.expanduser()
    grid_path = args.grid or root / "Data" / "metadata" / "output_grid_target.csv"
    grid = load_grid(grid_path)
    bbox = bbox_of(grid)
    print(f"Grid: {len(grid)} | Batas: {bbox[0]}..{bbox[2]} BT, {bbox[1]}..{bbox[3]} LS")
    if args.check_grid:
        return 0

    days = pd.date_range(start, end, freq="D")
    raw_root = root / "Data" / "raw" / "chla" / "MODISA_2022.0_4km_daily"
    processed = root / "Data" / "processed" / "chla"
    processed.mkdir(parents=True, exist_ok=True)
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}_{VERSION}"
    status_path = processed / f"download_status_{tag}.csv"

    discovered, earthaccess = {}, None
    if not args.offline:
        import earthaccess as ea
        earthaccess = ea
        print("Mencari file harian 4 km di katalog NASA...")
        discovered = discover(earthaccess, start, end, bbox)
        print(f"Tanggal dengan file: {len(discovered)}/{len(days)}")
        if args.catalog_only:
            return 0
        print("Login Earthdata di terminal. Password tidak disimpan oleh script.")
        auth = earthaccess.login(strategy="interactive", persist=False)
        if not auth.authenticated:
            raise RuntimeError("Login Earthdata gagal.")

    daily_records, status_records = [], []
    for number, day in enumerate(days, 1):
        filename = f"AQUA_MODIS.{day:%Y%m%d}.L3m.DAY.CHL.chlor_a.4km.nc"
        raw_dir = raw_root / f"{day:%Y}"
        path = raw_dir / filename
        entry = discovered.get(day)
        status = {
            "date": day, "filename": filename, "source_url": entry["source_url"] if entry else "",
            "concept_id": entry["concept_id"] if entry else "", "status": "", "local_file": str(path),
        }
        try:
            if not args.offline and entry is None:
                status["status"] = "not_in_catalog"
                daily_records.extend(aggregate_day(grid, day, source_status="not_in_catalog"))
            else:
                raw_dir.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    if args.offline:
                        raise FileNotFoundError(f"File offline belum ada: {path}")
                    earthaccess.download([entry["granule"]], local_path=raw_dir, threads=1, show_progress=False)
                if not path.is_file():
                    raise RuntimeError("Unduhan tidak menghasilkan file yang diharapkan.")
                subset = read_subset(path, bbox)
                daily_records.extend(aggregate_day(grid, day, subset))
                status.update(subset[3])
                status["status"] = "ok"
        except Exception as exc:
            status.update({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})
            status_records.append(status)
            atomic_csv(pd.DataFrame(status_records), status_path)
            raise RuntimeError(
                f"Berhenti pada {day:%Y-%m-%d}: {type(exc).__name__}: {exc}. "
                "File valid yang sudah ada tetap dipakai saat menjalankan ulang."
            ) from exc
        status_records.append(status)
        atomic_csv(pd.DataFrame(status_records), status_path)
        print(f"[{number}/{len(days)}] {day:%Y-%m-%d}: {status['status']}", flush=True)

    daily = pd.DataFrame(daily_records)
    daily["n_pixels_total"] = daily.n_pixels_total.astype("Int64")
    weekly = aggregate_weekly(daily)
    daily_path = processed / f"chla_daily_{tag}.csv"
    weekly_path = processed / f"chla_weekly_{tag}.csv"
    atomic_csv(daily, daily_path)
    atomic_csv(weekly, weekly_path)
    metadata = {
        "script_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION, "product": PRODUCT, "product_version": PRODUCT_VERSION,
        "doi": "10.5067/AQUA/MODIS/L3M/CHL/2022.0", "start": str(start.date()),
        "end": str(end.date()), "bbox": bbox, "grid_count": len(grid), "grid_file": str(grid_path),
        "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "source_resolution": "1/24 degree, nominal 4 km",
        "target_resolution_degree": 0.25,
        "spatial_method": "mean valid pixel centers per grid, weighted by cos(latitude)",
        "temporal_method": "equal-weight mean of available daily means, Monday-Sunday",
        "missing": "NaN; no interpolation; cloud/land missing is retained",
    }
    meta_path = processed / f"run_metadata_{tag}.json"
    temporary = meta_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(meta_path)
    print(f"Selesai: {len(daily)} baris harian dan {len(weekly)} baris mingguan.")
    print(f"Harian: {daily_path}\nMingguan: {weekly_path}")
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
