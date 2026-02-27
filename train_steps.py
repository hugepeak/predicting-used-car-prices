from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), sep=r"\s+")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    reg = pd.to_datetime(df["regDate"].astype(str).str.zfill(8), format="%Y%m%d", errors="coerce")
    crt = pd.to_datetime(df["creatDate"].astype(str).str.zfill(8), format="%Y%m%d", errors="coerce")
    df = df.copy()
    df["reg_year"] = reg.dt.year
    df["reg_month"] = reg.dt.month
    df["reg_day"] = reg.dt.day
    df["creat_year"] = crt.dt.year
    df["creat_month"] = crt.dt.month
    df["creat_day"] = crt.dt.day
    df["days_used"] = (crt - reg).dt.days
    return df


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["notRepairedDamage"] = pd.to_numeric(out["notRepairedDamage"], errors="coerce")
    out["power"] = pd.to_numeric(out["power"], errors="coerce")
    out["power"] = out["power"].clip(lower=0, upper=600)
    out["kilometer_bin"] = pd.to_numeric(out["kilometer"], errors="coerce").round(-1)
    out["power_bin"] = pd.qcut(
        out["power"].fillna(out["power"].median()),
        q=20,
        duplicates="drop",
    ).astype(str)
    out["model_bin"] = pd.qcut(
        pd.to_numeric(out["model"], errors="coerce").fillna(0),
        q=25,
        duplicates="drop",
    ).astype(str)
    return out


def build_group_stats(train_df: pd.DataFrame, train_y: pd.Series) -> dict[str, pd.DataFrame]:
    train_with_y = train_df.assign(price=train_y)
    out = {}
    for key in ["brand", "model"]:
        out[key] = train_with_y.groupby(key)["price"].agg(["mean", "max", "min", "std", "median"])
    return out


def apply_group_stats(df: pd.DataFrame, stats: dict[str, pd.DataFrame], train_y_mean: float) -> pd.DataFrame:
    out = df.copy()
    for key in ["brand", "model"]:
        if key not in out.columns or key not in stats:
            continue
        s = stats[key]
        prefix = f"{key}_price"
        out[f"{prefix}_mean"] = out[key].map(s["mean"]).fillna(train_y_mean)
        out[f"{prefix}_max"] = out[key].map(s["max"]).fillna(train_y_mean)
        out[f"{prefix}_min"] = out[key].map(s["min"]).fillna(train_y_mean)
        out[f"{prefix}_std"] = out[key].map(s["std"]).fillna(0)
        out[f"{prefix}_median"] = out[key].map(s["median"]).fillna(train_y_mean)
    return out


def clean_and_features(
    df: pd.DataFrame,
    target: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.Series | None]:
    df = df.copy()
    if target is not None:
        df = df.loc[target.index]
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    # If you want to experiment with removing regionCode, comment this line in/out.
    # if "regionCode" in df.columns:
    #     df = df.drop(columns=["regionCode"])

    df = parse_dates(df)
    df = add_base_features(df)

    if "kilometer_bin" in df.columns:
        df["kilometer"] = df["kilometer_bin"].astype(str)
    if "power_bin" in df.columns:
        df["power"] = df["power_bin"].astype(str)
    if "model_bin" in df.columns:
        df["model"] = df["model_bin"].astype(str)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = df[col].median()
        else:
            fill = "NA"
        df[col] = df[col].fillna(fill)

    df = df.drop(columns=[c for c in ["kilometer_bin", "power_bin", "model_bin"] if c in df.columns])
    if "SaleID" in df.columns:
        df = df.drop(columns=["SaleID"])
    return df, target


def infer_cat_cols(df: pd.DataFrame) -> list[str]:
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    for c in ["reg_year", "creat_year", "brand", "model", "bodyType", "fuelType", "gearbox"]:
        if c in df.columns and c not in cat_cols:
            cat_cols.append(c)
    return cat_cols


def run_quick(mode: str, sample_frac: float, seed: int = 42) -> None:
    train_raw = load_data("data/used_car_train_20200313.csv")
    train_raw = train_raw.loc[train_raw["seller"] != 1].copy()
    train_raw = train_raw.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
    y = train_raw["price"].copy()
    X = train_raw.drop(columns=["price"])

    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True
    )

    # build target-encoding statistics using training slice only
    group_stats = build_group_stats(X_train_raw, y_train)
    X_train, y_train = clean_and_features(X_train_raw, y_train)
    X_valid, y_valid = clean_and_features(X_valid_raw, y_valid)

    # apply group stats to both splits using only training-split statistics
    X_train = apply_group_stats(X_train, group_stats, float(y_train.mean()))
    X_valid = apply_group_stats(X_valid, group_stats, float(y_train.mean()))

    # keep same feature set
    X_train = X_train[X_valid.columns]
    if "price" in X_train.columns:
        X_train = X_train.drop(columns=["price"])
    if "price" in X_valid.columns:
        X_valid = X_valid.drop(columns=["price"])
    for c in X_train.columns:
        if c not in X_valid.columns:
            X_valid[c] = np.nan
    X_valid = X_valid[X_train.columns]

    # target box-cox to reduce long-tail effect
    pt = PowerTransformer(method="box-cox", standardize=False)
    y_train_t = pt.fit_transform(y_train.to_frame()).ravel()
    y_valid_t = pt.transform(y_valid.to_frame()).ravel()

    # cat feature index positions
    cat_cols = infer_cat_cols(X_train)
    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_valid[c] = X_valid[c].astype(str)
    cat_features = [X_train.columns.get_loc(c) for c in cat_cols]

    if mode == "quick":
        params = dict(
            depth=6,
            learning_rate=0.1,
            n_estimators=300,
            loss_function="MAE",
            random_seed=seed,
            eval_metric="MAE",
            verbose=False,
        )
    elif mode == "step2":
        params = dict(
            depth=8,
            learning_rate=0.08,
            n_estimators=600,
            loss_function="MAE",
            random_seed=seed,
            eval_metric="MAE",
            verbose=False,
        )
    else:
        params = dict(
            depth=9,
            learning_rate=0.05,
            n_estimators=1200,
            loss_function="MAE",
            random_seed=seed,
            eval_metric="MAE",
            verbose=False,
            early_stopping_rounds=120,
        )

    try:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train_t,
            eval_set=(X_valid, y_valid_t),
            cat_features=cat_features,
            verbose=False,
        )

        pred_t = model.predict(X_valid)
        pred = pt.inverse_transform(pred_t.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(y_valid, pred)
        print(f"{mode} | n={len(train_raw)} | MAE={mae:.2f}")
        return
    except Exception as exc:
        print(f"CatBoost training failed: {exc}")
        print("Please install or upgrade catboost first.")

    # no fallback in this file; use train_car_price.py for RandomForest baseline
    print("Only quick CatBoost validation is implemented in this file.")


def main() -> None:
    parser = ArgumentParser(description="Used-car price baseline in progressive mode")
    parser.add_argument(
        "--mode",
        choices=["quick", "step2", "full"],
        default="quick",
        help="quick: 5% data; step2: 15% data; full: all data",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frac_map = {"quick": 0.05, "step2": 0.15, "full": 1.0}
    run_quick(args.mode, sample_frac=frac_map[args.mode], seed=args.seed)


if __name__ == "__main__":
    main()
