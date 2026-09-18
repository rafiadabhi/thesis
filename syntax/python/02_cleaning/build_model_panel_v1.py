"""Gabungkan panel Chl-a, SST, dan angin bulanan pada grid yang sama.

Input:
  D:/thesis/data/processed/chla/chla_monthly_200301_202512_v1.csv
  D:/thesis/data/processed/sst/sst_monthly_200301_202512_v1.csv
  D:/thesis/data/processed/wind/wind_monthly_200301_202512_v1.csv

Output:
  D:/thesis/data/processed/model_dataset/selat_sunda_monthly_panel_200301_202512_v1.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v1"
KEYS = ["month", "grid_id"]


def read_panel(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV tidak ditemukan: {path}")
    frame = pd.read_csv(path, usecols=columns, parse_dates=["month"], dtype={"grid_id": str})
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Ada kombinasi month-grid_id duplikat pada {path.name}.")
    if frame[KEYS].isna().any().any():
        raise ValueError(f"Ada month atau grid_id kosong pada {path.name}.")
    return frame


def assert_coordinates(frame: pd.DataFrame, name: str) -> None:
    bad_lon = ~np.isclose(frame.lon, frame[f"lon_{name}"], atol=1e-9, rtol=0)
    bad_lat = ~np.isclose(frame.lat, frame[f"lat_{name}"], atol=1e-9, rtol=0)
    if (bad_lon | bad_lat).any():
        raise ValueError(f"Koordinat {name} tidak cocok dengan panel Chl-a.")


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
    args = parser.parse_args(argv)
    root = args.root.expanduser()

    chla_path = root / "data" / "processed" / "chla" / "chla_monthly_200301_202512_v1.csv"
    sst_path = root / "data" / "processed" / "sst" / "sst_monthly_200301_202512_v1.csv"
    wind_path = root / "data" / "processed" / "wind" / "wind_monthly_200301_202512_v1.csv"

    chla = read_panel(chla_path, [
        *KEYS, "lon", "lat", "chla_mg_m3", "n_valid_pixels", "n_pixels_total",
        "valid_pixel_fraction",
    ])
    sst = read_panel(sst_path, [
        *KEYS, "lon", "lat", "sst_c", "sst_anomaly_c",
    ]).rename(columns={"lon": "lon_sst", "lat": "lat_sst"})
    wind = read_panel(wind_path, [
        *KEYS, "lon", "lat", "u10_m_s", "v10_m_s", "wind_speed_m_s",
        "wind_direction_from_deg",
    ]).rename(columns={"lon": "lon_wind", "lat": "lat_wind"})

    panel = chla.merge(sst, on=KEYS, how="outer", validate="one_to_one", indicator="sst_merge")
    if not (panel.sst_merge == "both").all():
        raise ValueError("Kunci Chl-a dan SST tidak identik.")
    panel = panel.drop(columns="sst_merge")
    assert_coordinates(panel, "sst")
    panel = panel.merge(wind, on=KEYS, how="outer", validate="one_to_one", indicator="wind_merge")
    if not (panel.wind_merge == "both").all():
        raise ValueError("Kunci Chl-a dan angin tidak identik.")
    panel = panel.drop(columns="wind_merge")
    assert_coordinates(panel, "wind")

    panel = panel.drop(columns=["lon_sst", "lat_sst", "lon_wind", "lat_wind"])
    panel["chla_observed"] = panel.chla_mg_m3.notna()
    panel["predictors_complete"] = panel[["sst_c", "sst_anomaly_c", "u10_m_s", "v10_m_s"]].notna().all(axis=1)
    panel["model_complete"] = panel.chla_observed & panel.predictors_complete
    panel = panel.sort_values(["month", "grid_id"]).reset_index(drop=True)

    months = pd.DatetimeIndex(panel.month.unique())
    grids = pd.Index(panel.grid_id.unique())
    expected_months = pd.date_range(months.min(), months.max(), freq="MS")
    expected_rows = len(months) * len(grids)
    if not months.equals(expected_months) or len(panel) != expected_rows:
        raise ValueError("Panel gabungan tidak seimbang atau ada bulan yang hilang.")

    output_dir = root / "data" / "processed" / "model_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selat_sunda_monthly_panel_200301_202512_v1.csv"
    atomic_csv(panel, output_path)

    metadata = {
        "script_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": f"{months.min():%Y-%m} to {months.max():%Y-%m}",
        "grid_count": len(grids),
        "period_count": len(months),
        "row_count": len(panel),
        "model_complete_rows": int(panel.model_complete.sum()),
        "model_complete_fraction": float(panel.model_complete.mean()),
        "chla_missing_rows": int((~panel.chla_observed).sum()),
        "predictor_missing_rows": int((~panel.predictors_complete).sum()),
        "source_files": {
            "chla": str(chla_path),
            "sst": str(sst_path),
            "wind": str(wind_path),
        },
        "source_sha256": {
            "chla": hashlib.sha256(chla_path.read_bytes()).hexdigest(),
            "sst": hashlib.sha256(sst_path.read_bytes()).hexdigest(),
            "wind": hashlib.sha256(wind_path.read_bytes()).hexdigest(),
        },
        "target": "chla_mg_m3",
        "predictors": ["sst_c", "sst_anomaly_c", "u10_m_s", "v10_m_s"],
        "missingness_rule": "Chl-a missing values are retained; no imputation is applied in this step.",
    }
    metadata_path = output_dir / "model_panel_metadata_200301_202512_v1.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(metadata_path)

    print(f"Panel model: {len(panel)} baris = {len(grids)} grid x {len(months)} bulan")
    print(f"Target Chl-a teramati: {int(panel.chla_observed.sum())} ({panel.chla_observed.mean():.1%})")
    print(f"Prediktor lengkap: {int(panel.predictors_complete.sum())} ({panel.predictors_complete.mean():.1%})")
    print(f"Baris siap model: {int(panel.model_complete.sum())} ({panel.model_complete.mean():.1%})")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)
