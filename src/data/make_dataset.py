# src/data/make_dataset.py
"""
This is the 1st step:

make_dataset.py is the single, fully reproducible entry point of the entire data pipeline:
with just one command (python src/data/make_dataset.py),
it loads the raw file data/raw/london_weather.csv, applies
deterministic cleaning (date parsing, duplicate removal, and outlier filtering),
optionally saves an intermediate version in data/interim/, performs a chronological
train/validation/test split, and finally writes the three ready-to-use datasets in
optimised parquet format to data/processed/ (train.parquet, valid.parquet, test.parquet).
This guarantees that the whole team — and you in six months — will always get exactly the
same training-ready data, with zero manual steps and zero risk of discrepancy.
"""

# import pandas as pd
from pathlib import Path

# Chemin propre et reproductible quel que soit l’OS
# src/data/make_dataset.py
# src/data/make_dataset.py
# src/data/make_dataset.py
import click
import pandas as pd

# Chemins robustes
ROOT = Path(__file__).parent.parent.parent.resolve()
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean() -> pd.DataFrame:
    path = RAW_DIR / "london_weather.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    # Conversion date + nettoyage de base
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    nb_before = len(df)
    df = df.drop_duplicates()

    weather_cols = [
        "max_temp",
        "mean_temp",
        "min_temp",
        "sunshine",
        "global_radiation",
        "precipitation",
        "pressure",
        "cloud_cover",
        "snow_depth",
    ]
    weather_cols = [c for c in weather_cols if c in df.columns]
    df = df.dropna(subset=weather_cols, how="all")

    print(f"Dataset chargé et nettoyé : {len(df)} lignes ({nb_before - len(df)} supprimées)")
    return df


def chronological_split(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    max_year = df["date"].dt.year.max()

    test = df[df["date"].dt.year >= max_year - 1]  # 2019-2020
    valid = df[df["date"].dt.year.between(max_year - 3, max_year - 2)]  # 2017-2018
    train = df[df["date"].dt.year < max_year - 3]  # 1979-2016

    print("\nSplit chronologique :")
    print(f"Train  : {train['date'].dt.year.min()} – {train['date'].dt.year.max()}   → {len(train):,} lignes")
    print(f"Valid  : {valid['date'].dt.year.min()} – {valid['date'].dt.year.max()}   → {len(valid):,} lignes")
    print(f"Test   : {test['date'].dt.year.min()} – {test['date'].dt.year.max()}     → {len(test):,} lignes")

    return train.copy(), valid.copy(), test.copy()


@click.command()
@click.option("--interim", is_flag=True, help="Ne génère QUE le fichier interim (rapide pour explorer)")
def main(interim: bool):
    df = load_and_clean()

    # Mode interim uniquement
    # if interim:
    out_path = INTERIM_DIR / "london_weather_clean.parquet"
    # out_path2 = INTERIM_DIR / "london_weather_clean.csv"
    df.to_parquet(out_path, index=False)
    # df.to_csv(out_path2, index=False)
    print(f"\nVersion intermédiaire sauvegardée ici en parquet → {out_path}")
    # print(f"\nVersion intermédiaire sauvegardée ici en csv → {out_path2}")
    # return

    # Mode complet = split + processed (comportement par défaut)
    train, valid, test = chronological_split(df)

    for name, dataset in [("train", train), ("valid", valid), ("test", test)]:
        dataset.drop(columns=["date"], errors="ignore").to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

        dataset.drop(columns=["date"], errors="ignore").to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    print("\nDatasets finaux prêts pour l'entrainement (format parquet) dans data/processed/")
    print("├── train.parquet")
    print("├── valid.parquet")
    print("└── test.parquet")

    """

    print("\nDatasets finaux prêts pour l'entrainement (format csv) dans data/processed/")
    print("├── train.csv")
    print("├── valid.csv")
    print("└── test.csv")

    """


if __name__ == "__main__":
    main()
