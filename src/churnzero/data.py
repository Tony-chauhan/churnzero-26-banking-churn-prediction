from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_CANDIDATES = (
    "exited",
    "churn",
    "target",
    "is_churn",
    "ischurn",
    "attrition_flag",
    "attrited_customer",
    "label",
)

ID_CANDIDATES = (
    "customerid",
    "customer_id",
    "customer id",
    "clientnum",
    "client_num",
    "row_number",
    "rownumber",
    "id",
    "account_id",
)

TEXT_DROP_CANDIDATES = (
    "surname",
    "name",
    "firstname",
    "first_name",
    "lastname",
    "last_name",
    "email",
    "phone",
    "address",
)


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _canonical(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    by_canonical = {_canonical(col): col for col in df.columns}
    for candidate in candidates:
        key = _canonical(candidate)
        if key in by_canonical:
            return by_canonical[key]
    return None


def find_target_column(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Target column '{explicit}' not found in {list(df.columns)}")
        return explicit

    target = find_column(df, TARGET_CANDIDATES)
    if target:
        return target

    binary_columns = []
    for column in df.columns:
        unique = df[column].dropna().unique()
        if len(unique) == 2:
            binary_columns.append(column)

    if len(binary_columns) == 1:
        return binary_columns[0]

    raise ValueError(
        "Could not infer target column. Pass --target explicitly. "
        f"Binary candidates found: {binary_columns}"
    )


def find_id_column(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"ID column '{explicit}' not found in {list(df.columns)}")
        return explicit
    return find_column(df, ID_CANDIDATES)


def coerce_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = sorted(series.dropna().unique().tolist())
        if len(values) != 2:
            raise ValueError(f"Target must be binary. Found values: {values}")
        if set(values) == {0, 1}:
            return series.astype(int)
        return series.map({values[0]: 0, values[1]: 1}).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    positive = {
        "1",
        "yes",
        "y",
        "true",
        "t",
        "churn",
        "churned",
        "exited",
        "attrited customer",
        "attrited",
    }
    negative = {
        "0",
        "no",
        "n",
        "false",
        "f",
        "not churn",
        "not_churn",
        "retained",
        "existing customer",
        "active",
    }

    mapped = normalized.map(lambda value: 1 if value in positive else 0 if value in negative else np.nan)
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Could not map target values to binary classes: {unknown}")
    return mapped.astype(int)


def summarize_frame(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
