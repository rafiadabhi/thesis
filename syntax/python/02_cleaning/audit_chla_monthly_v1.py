"""Audit struktur dan kelengkapan panel Chl-a bulanan.

Input:
  D:/thesis/data/processed/chla/chla_monthly_200301_202512_v1.csv

Output:
  D:/thesis/data/processed/chla/chla_grid_availability_200301_202512_v1.csv
  D:/thesis/data/processed/chla/chla_month_availability_200301_202512_v1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v1"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", na_rep="")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\thesis"))
    parser.add_argument("--input", type=Path, help="CSV Chl-a bulanan.")
    parser.add_argument("--min-coverage", type=float, default=0.75,
                        help="Ambang laporan grid yang perlu ditinjau, 0 sampai 1.")
    args = parser.parse_args(argv)
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage harus berada pada rentang 0 sampai 1.")

    root = args.root.expanduser()
    processed = root / "data" / "processed" / "chla"
    input_path = args.input or processed / "chla_monthly_200301_202512_v1.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"CSV tidak ditemukan: {input_path}")

    data = pd.read_csv(input_path, parse_dates=["month"], dtype={"grid_id": str})
    required = {
        "month", "grid_id", "lon", "lat", "chla_mg_m3", "n_valid_pixels",
        "n_pixels_total", "valid_pixel_fraction", "source_status",
    }
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"Kolom belum lengkap: {missing_columns}")
    if data.empty:
        raise ValueError("CSV tidak memiliki baris.")
    if data[["month", "grid_id"]].isna().any().any():
        raise ValueError("Terdapat month atau grid_id kosong.")
    if data.duplicated(["month", "grid_id"]).any():
        raise ValueError("Ada kombinasi month-grid_id duplikat.")

    months = pd.DatetimeIndex(sorted(data.month.unique()))
    grids = pd.Index(sorted(data.grid_id.unique()))
    expected_months = pd.date_range(months.min(), months.max(), freq="MS")
    expected_rows = len(months) * len(grids)
    if not months.equals(expected_months):
        raise ValueError("Deret bulan tidak berurutan atau ada bulan yang hilang.")
    if len(data) != expected_rows:
        raise ValueError(
            f"Panel tidak seimbang: {len(data)} baris, seharusnya {expected_rows}."
        )

    data["available"] = data.chla_mg_m3.notna()
    grid_availability = (
        data.groupby("grid_id", as_index=False)
        .agg(
            lon=("lon", "first"),
            lat=("lat", "first"),
            n_months=("month", "size"),
            n_available=("available", "sum"),
            mean_chla_mg_m3=("chla_mg_m3", "mean"),
            median_chla_mg_m3=("chla_mg_m3", "median"),
            mean_valid_pixel_fraction=("valid_pixel_fraction", "mean"),
        )
    )
    grid_availability["n_missing"] = (
        grid_availability.n_months - grid_availability.n_available
    )
    grid_availability["availability_fraction"] = (
        grid_availability.n_available / grid_availability.n_months
    )
    grid_availability["review_flag"] = np.where(
        grid_availability.availability_fraction < args.min_coverage,
        "review", "keep",
    )
    grid_availability = grid_availability.sort_values("grid_id").reset_index(drop=True)

    month_availability = (
        data.groupby("month", as_index=False)
        .agg(
            n_grids=("grid_id", "size"),
            n_available=("available", "sum"),
            mean_chla_mg_m3=("chla_mg_m3", "mean"),
            mean_valid_pixel_fraction=("valid_pixel_fraction", "mean"),
        )
    )
    month_availability["n_missing"] = (
        month_availability.n_grids - month_availability.n_available
    )
    month_availability["availability_fraction"] = (
        month_availability.n_available / month_availability.n_grids
    )
    month_availability = month_availability.sort_values("month").reset_index(drop=True)

    grid_path = processed / "chla_grid_availability_200301_202512_v1.csv"
    month_path = processed / "chla_month_availability_200301_202512_v1.csv"
    atomic_csv(grid_availability, grid_path)
    atomic_csv(month_availability, month_path)

    source_status = data.source_status.value_counts(dropna=False).to_dict()
    total_available = int(data.available.sum())
    total_missing = int((~data.available).sum())
    print(f"Panel: {len(data)} baris = {len(grids)} grid x {len(months)} bulan")
    print(f"Periode: {months.min():%Y-%m} s.d. {months.max():%Y-%m}")
    print(f"Chl-a tersedia: {total_available} ({total_available / len(data):.1%})")
    print(f"Chl-a hilang: {total_missing} ({total_missing / len(data):.1%})")
    print(f"Status sumber: {source_status}")
    print(f"Grid di bawah ambang {args.min_coverage:.0%}: "
          f"{int((grid_availability.review_flag == 'review').sum())}")
    print(f"Output grid: {grid_path}")
    print(f"Output bulan: {month_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)
