# =========================
# RandomForest - Entraînement uniquement à partir des .npy avec MLflow
# + Sauvegarde locale MLOps
# =========================

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Définir la racine du projet de manière robuste
ROOT = Path(__file__).resolve().parent.parent.parent  # src/models → src → projet racine

# Charger params.yaml depuis la racine
params_path = ROOT / "params.yaml"
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

FEAT_DIR = ROOT / "data" / "features" / "random_forest"

X_train = np.load(FEAT_DIR / "X_train.npy")
y_train = np.load(FEAT_DIR / "y_train.npy").ravel()

print("Shapes:", "X_train:", X_train.shape, "| y_train:", y_train.shape)

# =========================
# MLflow : traçabilité de l’entraînement
# =========================
mlflow.sklearn.autolog()
mlflow.set_experiment("weather_mean_temp_models")

# Récupération des paramètres du RandomForest
rf_params = params["random_forest"]

# Pipeline RandomForest
rf_pipeline = Pipeline(steps=[("rf", RandomForestRegressor(**rf_params))])

# =========================
# Entraînement + Sauvegarde locale
# =========================
MODELS_DIR = (ROOT / "models" / "random_forest" / "Production").resolve()
MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

with mlflow.start_run(run_name="RandomForest_train_only_from_npy") as run:
    rf_pipeline.fit(X_train, y_train)

    run_id = run.info.run_id

    # (Optionnel mais recommandé) Log explicite des params (autolog le fait souvent, mais on sécurise)
    mlflow.log_params(rf_params)
    mlflow.log_param("features_dir", str(FEAT_DIR.resolve()))

    # ✅ Sauvegarde locale du modèle (sortie stable pour DVC)
    joblib.dump(rf_pipeline, MODEL_PATH)

    # ✅ Sauvegarde des métadonnées (traçabilité)
    model_info = {
        "run_id": run_id,
        "experiment_name": "weather_mean_temp_models",
        "model_type": "RandomForestRegressor",
        "features_dir": str(FEAT_DIR.resolve()),
        "model_path": str(MODEL_PATH),
        "params": rf_params,
    }
    MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    # (Optionnel) log aussi ces fichiers comme artefacts pour tout retrouver dans MLflow
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="local_model_copy")
    mlflow.log_artifact(str(MODEL_INFO_PATH), artifact_path="local_model_copy")

print("✅ Entraînement terminé.")
print(f"✅ Modèle sauvegardé localement : {MODEL_PATH}")
print(f"✅ Infos modèle sauvegardées    : {MODEL_INFO_PATH}")
