#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SplitData = Tuple[np.ndarray, np.ndarray]


# -----------------------------
# Utilitaires data / metrics
# -----------------------------
def load_npy_split(feat_dir: Path) -> Tuple[SplitData, SplitData, SplitData]:
    """Charge X_train/X_valid/X_test et y_train/y_valid/y_test depuis un dossier."""
    try:
        X_train = np.load(feat_dir / "X_train.npy")
        X_valid = np.load(feat_dir / "X_valid.npy")
        X_test = np.load(feat_dir / "X_test.npy")

        y_train = np.load(feat_dir / "y_train.npy").squeeze()
        y_valid = np.load(feat_dir / "y_valid.npy").squeeze()
        y_test = np.load(feat_dir / "y_test.npy").squeeze()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Fichier .npy manquant dans {feat_dir}: {e}") from e

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X)
    return {
        "rmse": rmse(y, y_pred),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }


# -----------------------------
# Chargement modèle
# -----------------------------
def load_model_local(model_path: Path) -> BaseEstimator:
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle local introuvable: {model_path}")
    print(f"📦 Chargement du modèle local: {model_path}")
    return joblib.load(model_path)


def load_model_from_mlflow(model_uri: str) -> BaseEstimator:
    print(f"🚀 Chargement du modèle depuis MLflow URI: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def try_read_train_run_id(model_info_path: Path) -> Optional[str]:
    """
    Optionnel : si model_info.json existe (créé par train_model.py),
    on récupère run_id pour relier éval ↔ train.
    """
    if not model_info_path.exists():
        return None
    try:
        data = json.loads(model_info_path.read_text(encoding="utf-8"))
        return data.get("run_id")
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Évaluer un modèle et enregistrer métriques/artefacts dans MLflow.")

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/features/random_forest"),
        help="Dossier contenant X_*.npy et y_*.npy (ex: data/features/random_forest)",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="weather_mean_temp_models", help="Nom de l'expérience MLflow"
    )
    parser.add_argument(
        "--run-name", type=str, default="RandomForest_evaluate", help="Nom du run MLflow pour l'évaluation"
    )

    # Modèle : local ou MLflow
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-path", type=Path, help="Chemin local du modèle (ex: models/random_forest/Production/model.pkl)"
    )
    group.add_argument(
        "--model-uri", type=str, help="URI MLflow du modèle (ex: runs:/<run_id>/model ou models:/Nom/Production)"
    )

    # Sorties / artefacts
    parser.add_argument(
        "--metrics-path", type=Path, default=Path("metrics.json"), help="Fichier JSON des métriques (utile pour DVC)."
    )

    # Option : log modèle dans ce run d'éval (souvent inutile si train autolog)
    parser.add_argument(
        "--log-model",
        action="store_true",
        help="Si activé, loggue aussi le modèle dans CE run d'évaluation (sinon métriques seulement).",
    )

    # Option : copie locale du modèle évalué (rarement nécessaire, mais possible)
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Si activé, sauvegarde aussi une copie locale du modèle évalué via joblib.",
    )
    parser.add_argument(
        "--save-local-dir",
        type=Path,
        default=Path("models/random_forest/Production"),
        help="Dossier local pour la copie du modèle évalué.",
    )

    # Option : relier explicitement à un run d'entraînement
    parser.add_argument("--train-run-id", type=str, default=None, help="Run ID du training correspondant (optionnel).")

    args = parser.parse_args()

    feat_dir = args.features_dir.resolve()
    metrics_path = args.metrics_path.resolve()
    save_local_dir = args.save_local_dir.resolve()

    print(f"📁 Features dir: {feat_dir}")

    # 1) Charger datasets
    print("🔄 Chargement des datasets...")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_npy_split(feat_dir)
    print(f"✅ Datasets chargés. Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # 2) Charger modèle
    model_source: str
    inferred_train_run_id: Optional[str] = None

    if args.model_path:
        model_path = args.model_path.resolve()
        model = load_model_local(model_path)
        model_source = f"local:{model_path}"

        # Si on évalue le modèle sauvegardé par train_model.py, on lit model_info.json
        # (même dossier que model.pkl)
        model_info_path = model_path.parent / "model_info.json"
        inferred_train_run_id = try_read_train_run_id(model_info_path)

    else:
        model = load_model_from_mlflow(args.model_uri)
        model_source = f"mlflow:{args.model_uri}"

    # train_run_id : priorité à l’argument, sinon on infère depuis model_info.json
    train_run_id = args.train_run_id or inferred_train_run_id

    # 3) Évaluer
    print("🧪 Évaluation du modèle...")
    metrics_train = evaluate(model, X_train, y_train)
    metrics_valid = evaluate(model, X_valid, y_valid)
    metrics_test = evaluate(model, X_test, y_test)

    out_metrics = {"train": metrics_train, "valid": metrics_valid, "test": metrics_test}
    print("✅ Évaluation terminée.")

    # 4) MLflow logging
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print(f"🟢 Run MLflow évaluation: {run_id}")

        # Params
        mlflow.log_param("model_source", model_source)
        mlflow.log_param("features_dir", str(feat_dir))
        if train_run_id:
            mlflow.log_param("train_run_id", train_run_id)

        # Tags (pratique pour filtrer dans MLflow UI)
        mlflow.set_tag("stage", "evaluate")
        mlflow.set_tag("model_source", model_source)

        # Métriques
        for split_name, split_metrics in out_metrics.items():
            for k, v in split_metrics.items():
                mlflow.log_metric(f"{split_name}_{k}", v)

        # metrics.json (DVC) + artefact MLflow
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation_metrics")

        # Optionnel : re-log du modèle dans le run d'éval
        if args.log_model:
            mlflow.sklearn.log_model(model, artifact_path="model_evaluated")
            print("📦 Modèle loggué dans MLflow (artifact_path='model_evaluated').")

        # Optionnel : copie locale du modèle évalué
        if args.save_local:
            save_local_dir.mkdir(parents=True, exist_ok=True)
            local_path = save_local_dir / "model_evaluated.pkl"
            joblib.dump(model, local_path)
            mlflow.log_artifact(str(local_path), artifact_path="local_copy")
            print(f"💾 Copie locale sauvegardée: {local_path}")

    # 5) Synthèse console
    print("\n" + "=" * 30)
    print("=== Synthèse des Résultats ===")
    print(f"Train: RMSE={metrics_train['rmse']:.4f}, MAE={metrics_train['mae']:.4f}, R2={metrics_train['r2']:.4f}")
    print(f"Valid: RMSE={metrics_valid['rmse']:.4f}, MAE={metrics_valid['mae']:.4f}, R2={metrics_valid['r2']:.4f}")
    print(f"Test : RMSE={metrics_test['rmse']:.4f}, MAE={metrics_test['mae']:.4f}, R2={metrics_test['r2']:.4f}")
    print("=" * 30)
    print("✅ Évaluation terminée et artefacts loggés dans MLflow.")


if __name__ == "__main__":
    main()
