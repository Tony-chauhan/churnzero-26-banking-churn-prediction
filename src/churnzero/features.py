from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnzero.data import ID_CANDIDATES, TEXT_DROP_CANDIDATES, _canonical


def choose_feature_columns(
    df: pd.DataFrame,
    target_column: str | None,
    id_column: str | None,
    max_text_cardinality_ratio: float = 0.5,
) -> tuple[list[str], list[str]]:
    dropped: set[str] = set()
    if target_column:
        dropped.add(target_column)
    if id_column:
        dropped.add(id_column)

    for column in df.columns:
        canonical = _canonical(column)
        if canonical in {_canonical(name) for name in ID_CANDIDATES}:
            dropped.add(column)
            continue
        if canonical in {_canonical(name) for name in TEXT_DROP_CANDIDATES}:
            dropped.add(column)
            continue
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            non_null = max(int(df[column].notna().sum()), 1)
            cardinality_ratio = df[column].nunique(dropna=True) / non_null
            if cardinality_ratio > max_text_cardinality_ratio and df[column].nunique(dropna=True) > 30:
                dropped.add(column)

    feature_columns = [column for column in df.columns if column not in dropped]
    return feature_columns, [column for column in df.columns if column in dropped]


def split_feature_types(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_features = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
