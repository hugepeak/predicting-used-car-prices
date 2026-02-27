from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


def read_car_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+")


def infer_categorical_cols(df: pd.DataFrame, min_frac: float = 0.05, max_unique_ratio: float = 0.01) -> list[str]:
    """
    将字符串列和低基数数值列视为类别特征，提升树模型对离散字段的表达能力。
    """
    cat_cols: list[str] = [c for c in df.columns if df[c].dtype == object]
    for c in df.columns:
        if c in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nunique = df[c].nunique(dropna=True)
            ratio = nunique / max(len(df), 1)
            if nunique <= 200 and ratio <= max_unique_ratio:
                cat_cols.append(c)
    return cat_cols


def build_preprocessor(
    X: pd.DataFrame,
    categorical_cols: Iterable[str],
) -> ColumnTransformer:
    categorical_cols = list(categorical_cols)
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    categorical_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )


def train_model(
    train_path: str | Path = "data/used_car_train_20200313.csv",
    test_path: str | None = None,
    output_pred: str | None = None,
) -> None:
    train_df = read_car_data(Path(train_path))

    # SaleID 对预测无帮助，通常直接剔除
    if "SaleID" in train_df.columns:
        train_df = train_df.drop(columns=["SaleID"])

    target_col = "price"
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # 优先保留字符串类别特征，提升模型对离散取值的刻画
    cat_cols = infer_categorical_cols(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = build_preprocessor(X_train, cat_cols)

    # 以 MAE 优化方向为目标，并通过树模型处理高维类别编码后的特征
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=16,
        n_jobs=-1,
        random_state=42,
    )

    # 优先尝试 CatBoost（通常在这类混合离散特征效果更好）
    try:
        from catboost import CatBoostRegressor

        cat_feature_indexes = [i for i, c in enumerate(X_train.columns) if c in cat_cols]
        X_train_cb = X_train.copy()
        X_valid_cb = X_valid.copy()
        for col in cat_cols:
            X_train_cb[col] = X_train_cb[col].fillna("NA").astype(str)
            X_valid_cb[col] = X_valid_cb[col].fillna("NA").astype(str)

        cb_model = CatBoostRegressor(
            depth=9,
            learning_rate=0.05,
            n_estimators=2500,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=42,
            verbose=200,
        )
        cb_model.fit(
            X_train_cb,
            y_train,
            eval_set=(X_valid_cb, y_valid),
            cat_features=cat_feature_indexes,
            early_stopping_rounds=200,
        )

        pred_valid = cb_model.predict(X_valid_cb)
        mae = mean_absolute_error(y_valid, pred_valid)
        print(f"CatBoost valid MAE: {mae:.4f}")

        if output_pred:
            test_df = read_car_data(Path(test_path))
            sale_ids = test_df["SaleID"].copy()
            if "SaleID" in test_df.columns:
                test_df = test_df.drop(columns=["SaleID"])
            for c in cat_cols:
                if c in test_df.columns:
                    test_df[c] = test_df[c].fillna("NA").astype(str)
            preds = cb_model.predict(test_df)
            submit = pd.DataFrame({"SaleID": sale_ids, "price": np.maximum(0, preds)})
            pd.DataFrame({"SaleID": sale_ids, "price": np.maximum(0, preds)}).to_csv(output_pred, index=False)
            print(f"已导出预测文件: {output_pred}")
        return

    except Exception:
        print("CatBoost 未安装或训练失败，回退到 sklearn RandomForestRegressor。")

    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred_valid = pipe.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred_valid)
    print(f"RandomForest valid MAE: {mae:.4f}")

    if output_pred:
        test_df = read_car_data(Path(test_path))
        sale_ids = test_df["SaleID"] if "SaleID" in test_df.columns else None
        if "SaleID" in test_df.columns:
            test_df = test_df.drop(columns=["SaleID"])
        test_pred = np.maximum(0, pipe.predict(test_df))
        if sale_ids is not None:
            pd.DataFrame({"SaleID": sale_ids, "price": test_pred}).to_csv(output_pred, index=False)
            print(f"已导出预测文件: {output_pred}")


if __name__ == "__main__":
    train_model(
        train_path="data/used_car_train_20200313.csv",
        test_path="data/used_car_testB_20200421.csv",
        output_pred="outputs/catboost_submit.csv",
    )
