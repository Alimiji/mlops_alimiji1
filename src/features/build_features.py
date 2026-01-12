import os
from pathlib import Path

import numpy as np
import pandas as pd

# ============================
# 1. Configuration générale
# ============================

ROOT = Path(__file__).parent.parent.parent.resolve()

# Ce code permet de crééer les partitions de dataset pour le train, la validation et le test

DATA_PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"

TRAIN_PATH = DATA_PROCESSED_DIR / "train.parquet"
VALID_PATH = DATA_PROCESSED_DIR / "valid.parquet"
TEST_PATH = DATA_PROCESSED_DIR / "test.parquet"

# Colonne cible (comme dans ton notebook : mean_temp)
TARGET_COL = "mean_temp"

# ============================
# 2. Définition des features par modèle
# ============================

"""

# LinearRegression : sensible aux features, peu de colonnes
LR_FEATURES = ["min_temp", "global_radiation"]

# DecisionTreeRegressor : quelques features importantes
DTR_FEATURES = ["min_temp", "max_temp", "global_radiation", "sunshine"]


On se limitera seulement au RandomForest Regressor car le meilleur modèle


"""

# RandomForestRegressor : toutes les features utiles
RFR_FEATURES = [
    "min_temp",
    "max_temp",
    "global_radiation",
    "sunshine",
    "cloud_cover",
    "precipitation",
    "pressure",
    "snow_depth",
]


# ============================
# 3. Fonctions utilitaires
# ============================


def load_splits():
    """Charge les splits train/valid/test au format parquet."""
    if not TRAIN_PATH.exists() or not VALID_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Les fichiers train/valid/test parquet sont introuvables dans {DATA_PROCESSED_DIR}.\n"
            f"Attendu : {TRAIN_PATH.name}, {VALID_PATH.name}, {TEST_PATH.name}"
        )

    train_df = pd.read_parquet(TRAIN_PATH)
    valid_df = pd.read_parquet(VALID_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print("[INFO] Splits parquet chargés depuis data/processed/")
    print("       train shape:", train_df.shape)
    print("       valid shape:", valid_df.shape)
    print("       test  shape:", test_df.shape)

    return train_df, valid_df, test_df


def clean_target(train_df, valid_df, test_df, target_col=TARGET_COL):
    """
    Gère la cible (mean_temp) :
    - Calcule la moyenne sur le train
    - Impute les valeurs manquantes dans train/valid/test avec cette moyenne
    """
    if target_col not in train_df.columns:
        raise KeyError(f"La colonne cible '{target_col}' n'existe pas dans train.parquet")

    target_mean = train_df[target_col].mean()
    print(f"[INFO] Moyenne de la cible '{target_col}' sur le train : {target_mean:.4f}")

    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if target_col not in df.columns:
            raise KeyError(f"La colonne cible '{target_col}' n'existe pas dans {name}.parquet")
        n_missing = df[target_col].isna().sum()
        if n_missing > 0:
            print(f"[WARN] {n_missing} valeurs manquantes dans '{target_col}' pour {name} → imputées par la moyenne.")
        df[target_col] = df[target_col].fillna(target_mean)

    return train_df, valid_df, test_df


def select_features(df, feature_list):
    """
    Sélectionne les colonnes demandées mais ne garde que celles
    qui existent réellement dans le DataFrame.
    """
    existing_features = [col for col in feature_list if col in df.columns]

    if not existing_features:
        raise ValueError(
            f"Aucune des features demandées {feature_list} n'est présente dans le DataFrame.\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )

    X = df[existing_features].values
    return X, existing_features


def save_npy_features(model_name, X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Sauvegarde les matrices X/y au format .npy dans data/features/<model_name>/.
    Noms de fichiers identiques pour tous les modèles, mais rangés dans des dossiers différents.
    """
    model_dir = FEATURES_DIR / model_name
    os.makedirs(model_dir, exist_ok=True)

    np.save(model_dir / "X_train.npy", X_train)
    np.save(model_dir / "y_train.npy", y_train)
    np.save(model_dir / "X_valid.npy", X_valid)
    np.save(model_dir / "y_valid.npy", y_valid)
    np.save(model_dir / "X_test.npy", X_test)
    np.save(model_dir / "y_test.npy", y_test)

    print(f"[OK] Features sauvegardées pour '{model_name}' dans {model_dir}")


def build_features_for_model(model_name, feature_list, train_df, valid_df, test_df):
    """
    Construit X_train, X_valid, X_test et y_* pour UN modèle donné,
    en utilisant SA liste de features, puis les sauvegarde en .npy.
    """
    # === X : features spécifiques au modèle ===
    X_train, used_cols = select_features(train_df, feature_list)
    X_valid, _ = select_features(valid_df, feature_list)
    X_test, _ = select_features(test_df, feature_list)

    # === y : même cible pour tous les modèles ===
    y_train = train_df[TARGET_COL].values
    y_valid = valid_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    print(f"\n=== Modèle : {model_name} ===")
    print(f"Features utilisées : {used_cols}")
    print(f"Shapes -> X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")
    print(f"          y_train: {y_train.shape}, y_valid: {y_valid.shape}, y_test: {y_test.shape}")

    # Sauvegarde
    save_npy_features(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )


# ============================
# 4. Point d'entrée principal
# ============================


def main():
    print("============== BUILD FEATURES ==============")

    # Chargement des splits
    train_df, valid_df, test_df = load_splits()

    # Nettoyage / imputation de la cible
    train_df, valid_df, test_df = clean_target(train_df, valid_df, test_df, target_col=TARGET_COL)

    # Création du dossier racine des features
    os.makedirs(FEATURES_DIR, exist_ok=True)

    """

    # 1) Features pour Linear Regression
    build_features_for_model(
        model_name="linear_regression",
        feature_list=LR_FEATURES,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    # 2) Features pour Decision Tree
    build_features_for_model(
        model_name="decision_tree",
        feature_list=DTR_FEATURES,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    On se limitera seulement au modèle Random Forest car c'est le modèle ayant le meilleur score

    """

    # 3) Features pour Random Forest
    build_features_for_model(
        model_name="random_forest",
        feature_list=RFR_FEATURES,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    print("\n✅ Génération des features terminée.")
    print(f"   Dossier de sortie : {FEATURES_DIR.absolute()}")


if __name__ == "__main__":
    main()
