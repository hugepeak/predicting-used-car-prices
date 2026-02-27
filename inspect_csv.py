from pathlib import Path

import pandas as pd


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "data" / "used_car_train_20200313.csv"

    df = pd.read_csv(csv_path, sep=r"\s+")

    print("字段名：")
    print(df.columns.tolist())
    print("\n前5行：")
    print(df.head())


if __name__ == "__main__":
    main()
