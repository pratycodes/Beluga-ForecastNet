"""Dataset adapter for the GEFCom2014 hourly load track."""

from __future__ import annotations

import logging
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CELL_REF_RE = re.compile(r"([A-Z]+)([0-9]+)")
EXCEL_EPOCH = datetime(1899, 12, 30)
DEFAULT_RECENT_YEARS = 1


def build_gefcom_load_features(
    dataset_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    recent_years: int | None = DEFAULT_RECENT_YEARS,
) -> pd.DataFrame:
    """Build a compact, non-leaky feature matrix for GEFCom2014 load forecasting."""

    workbook_path = resolve_gefcom_load_path(dataset_path)
    hourly = load_gefcom_hourly_load(workbook_path)
    filtered = filter_gefcom_date_range(
        hourly,
        start_date=start_date,
        end_date=end_date,
        recent_years=recent_years,
    )
    features = _build_hourly_features(filtered)
    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    metadata = {
        "dataset_name": "GEFCom2014-E load",
        "dataset_path": str(workbook_path),
        "rows_before_missing_load_drop": int(hourly.attrs.get("raw_row_count", len(hourly))),
        "missing_load_rows": int(hourly.attrs.get("missing_load_count", 0)),
        "known_load_rows": int(len(hourly)),
        "selected_rows": int(len(filtered)),
        "start_timestamp": str(filtered["timestamp"].iloc[0]),
        "end_timestamp": str(filtered["timestamp"].iloc[-1]),
        "raw_series_count": 2,
        "target_series_count": 1,
        "exogenous_series_count": 1,
        "target_series_names": ["load"],
        "exogenous_series_names": ["temperature"],
        "recent_years": recent_years,
        "start_date": start_date,
        "end_date": end_date,
    }
    features.attrs["metadata"] = metadata
    logger.info(
        "Built GEFCom2014 load feature matrix shape=%s date_range=%s -> %s raw_series=%s target_series=%s exogenous_series=%s source=%s",
        features.shape,
        metadata["start_timestamp"],
        metadata["end_timestamp"],
        metadata["raw_series_count"],
        metadata["target_series_names"],
        metadata["exogenous_series_names"],
        workbook_path,
    )
    return features.astype(float)


def resolve_gefcom_load_path(dataset_path: str | Path) -> Path:
    """Resolve the GEFCom2014 electricity workbook from a folder or direct path."""

    path = Path(dataset_path)
    candidates = []
    if path.is_file():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path / "GEFCom2014-E_V2" / "GEFCom2014-E.xlsx",
                path / "GEFCom2014-E.xlsx",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"GEFCom2014 load workbook not found under: {path}")


def load_gefcom_hourly_load(path: str | Path) -> pd.DataFrame:
    """Load known hourly load and temperature rows from the GEFCom2014 workbook."""

    rows = read_xlsx_rows(path)
    if len(rows) < 2:
        raise ValueError(f"GEFCom2014 workbook has no data rows: {path}")
    header = rows[0]
    expected = ["Date", "Hour", "load", "T"]
    if header[: len(expected)] != expected:
        raise ValueError(f"Unexpected GEFCom2014 hourly header: {header}")

    records: list[dict[str, Any]] = []
    missing_load_count = 0
    for row in rows[1:]:
        if len(row) < len(expected):
            raise ValueError(f"Malformed GEFCom2014 row: {row}")
        timestamp = _excel_date_hour_to_timestamp(row[0], row[1])
        load_value = str(row[2]).strip()
        if load_value == "":
            missing_load_count += 1
            continue
        records.append(
            {
                "timestamp": timestamp,
                "load": float(load_value),
                "temperature": float(row[3]),
            }
        )

    frame = pd.DataFrame.from_records(records).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"GEFCom2014 workbook has no known load rows: {path}")
    _validate_hourly_continuity(frame)
    frame.attrs["raw_row_count"] = len(rows) - 1
    frame.attrs["missing_load_count"] = missing_load_count
    logger.info(
        "Loaded GEFCom2014 load workbook known_rows=%s missing_load_rows=%s range=%s -> %s",
        len(frame),
        missing_load_count,
        frame["timestamp"].iloc[0],
        frame["timestamp"].iloc[-1],
    )
    return frame


def read_xlsx_rows(path: str | Path) -> list[list[str]]:
    """Read the first worksheet from a simple .xlsx file without openpyxl."""

    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        rows: list[list[str]] = []
        with archive.open("xl/worksheets/sheet1.xml") as sheet:
            for _, element in ElementTree.iterparse(sheet, events=("end",)):
                if element.tag != XLSX_NS + "row":
                    continue
                values: dict[int, str] = {}
                max_index = -1
                for cell in element.findall(XLSX_NS + "c"):
                    reference = cell.attrib.get("r", "")
                    if not reference:
                        continue
                    column_index = _column_index(reference)
                    max_index = max(max_index, column_index)
                    value_node = cell.find(XLSX_NS + "v")
                    if value_node is None or value_node.text is None:
                        value = ""
                    elif cell.attrib.get("t") == "s":
                        value = shared_strings[int(value_node.text)]
                    else:
                        value = value_node.text
                    values[column_index] = value
                if max_index >= 0:
                    rows.append([values.get(index, "") for index in range(max_index + 1)])
                element.clear()
    return rows


def filter_gefcom_date_range(
    frame: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    recent_years: int | None = DEFAULT_RECENT_YEARS,
) -> pd.DataFrame:
    """Filter GEFCom rows to a compact training interval."""

    timestamp = pd.to_datetime(frame["timestamp"])
    start = _parse_boundary(start_date, is_end=False)
    end = _parse_boundary(end_date, is_end=True)

    if start is None and end is None and recent_years is not None:
        if recent_years < 1:
            raise ValueError("recent_years must be at least 1 when provided")
        last_year = int(timestamp.max().year)
        start = pd.Timestamp(year=last_year - recent_years + 1, month=1, day=1)
        end = pd.Timestamp(timestamp.max())

    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= timestamp >= start
    if end is not None:
        mask &= timestamp <= end

    filtered = frame.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            "GEFCom2014 date filter produced no rows "
            f"start_date={start_date!r} end_date={end_date!r} recent_years={recent_years!r}"
        )
    _validate_hourly_continuity(filtered)
    return filtered


def _build_hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(hourly["timestamp"])
    load = hourly["load"].astype(float)
    temperature = hourly["temperature"].astype(float)
    features = pd.DataFrame(index=hourly.index)

    features["load"] = load
    features["temperature"] = temperature
    features["temperature_squared"] = temperature**2
    features["heating_degree"] = np.maximum(65.0 - temperature, 0.0)
    features["cooling_degree"] = np.maximum(temperature - 65.0, 0.0)
    features["load_temperature_product"] = load * temperature

    hour = timestamp.dt.hour
    dayofweek = timestamp.dt.dayofweek
    month = timestamp.dt.month
    dayofyear = timestamp.dt.dayofyear
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    features["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    features["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 366)
    features["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 366)
    features["weekday_indicator"] = (dayofweek < 5).astype(int)
    features["weekend_indicator"] = (dayofweek >= 5).astype(int)

    for lag in (1, 2, 3, 6, 12, 24, 48, 168):
        features[f"load_lag_{lag}"] = load.shift(lag).fillna(load.iloc[0])
    for lag in (1, 3, 6, 12, 24, 168):
        features[f"temperature_lag_{lag}"] = temperature.shift(lag).fillna(temperature.iloc[0])

    features["load_diff_1"] = load.diff().fillna(0.0)
    features["load_diff_24"] = (load - load.shift(24)).fillna(0.0)
    features["load_diff_168"] = (load - load.shift(168)).fillna(0.0)
    features["temperature_diff_1"] = temperature.diff().fillna(0.0)
    features["temperature_diff_24"] = (temperature - temperature.shift(24)).fillna(0.0)

    for window in (3, 6, 12, 24, 168):
        rolling_load = load.rolling(window=window, min_periods=1)
        features[f"load_roll_mean_{window}"] = rolling_load.mean()
        features[f"load_roll_std_{window}"] = rolling_load.std().fillna(0.0)
        features[f"load_roll_min_{window}"] = rolling_load.min()
        features[f"load_roll_max_{window}"] = rolling_load.max()

    for window in (3, 24, 168):
        rolling_temperature = temperature.rolling(window=window, min_periods=1)
        features[f"temperature_roll_mean_{window}"] = rolling_temperature.mean()
        features[f"temperature_roll_std_{window}"] = rolling_temperature.std().fillna(0.0)

    return features


def _excel_date_hour_to_timestamp(date_value: str, hour_value: str) -> pd.Timestamp:
    day = EXCEL_EPOCH + timedelta(days=float(date_value))
    hour = int(float(hour_value))
    if hour < 1 or hour > 24:
        raise ValueError(f"GEFCom2014 hour must be in 1..24, got {hour_value!r}")
    return pd.Timestamp(day + timedelta(hours=hour - 1))


def _parse_boundary(value: str | None, is_end: bool) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    timestamp = pd.Timestamp(value)
    if is_end and timestamp == timestamp.normalize():
        timestamp = timestamp + pd.Timedelta(hours=23)
    return timestamp


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall(XLSX_NS + "si"):
        strings.append("".join(node.text or "" for node in item.iter(XLSX_NS + "t")))
    return strings


def _column_index(cell_reference: str) -> int:
    match = CELL_REF_RE.match(cell_reference)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference}")
    letters = match.group(1)
    index = 0
    for character in letters:
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _validate_hourly_continuity(frame: pd.DataFrame) -> None:
    timestamp = pd.to_datetime(frame["timestamp"])
    if timestamp.duplicated().any():
        duplicates = timestamp[timestamp.duplicated()].head().tolist()
        raise ValueError(f"Duplicate GEFCom2014 timestamps found: {duplicates}")
    differences = timestamp.diff().dropna()
    bad_differences = differences[differences != pd.Timedelta(hours=1)]
    if not bad_differences.empty:
        first_bad_index = int(bad_differences.index[0])
        raise ValueError(
            "GEFCom2014 timestamps are not continuous hourly at "
            f"{timestamp.iloc[first_bad_index - 1]} -> {timestamp.iloc[first_bad_index]}"
        )
