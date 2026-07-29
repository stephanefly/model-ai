"""Entraîne et compare deux modèles de probabilité de signature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import prepare_features, select_feature_columns


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path) -> pd.DataFrame:
    # sep=None reconnaît les exports séparés par virgule ou point-virgule.
    return pd.read_csv(path, sep=None, engine="python")


def make_preprocessor(numerics: list[str], categoricals: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("missing", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("missing", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numerics),
            ("categorical", categorical_pipe, categoricals),
        ],
        remainder="drop",
    )


def make_model(
    model_name: str,
    numerics: list[str],
    categoricals: list[str],
    seed: int,
    calibration_folds: int,
) -> CalibratedClassifierCV:
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.7,
            random_state=seed,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    pipeline = Pipeline(
        [
            ("preparation", make_preprocessor(numerics, categoricals)),
            ("classifier", classifier),
        ]
    )
    return CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=calibration_folds,
        n_jobs=-1,
    )


def score_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probability = model.predict_proba(x_test)[:, 1]
    return {
        "brier_score": float(brier_score_loss(y_test, probability)),
        "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "average_precision": float(average_precision_score(y_test, probability)),
        "validation_rows": int(len(y_test)),
        "validation_signature_rate": float(y_test.mean()),
        "mean_predicted_probability": float(probability.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="CSV produit par la requête SQL")
    parser.add_argument("--output-dir", default=Path("models"), type=Path)
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("config.json"),
        type=Path,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw = read_csv(args.data)

    required = {
        "request_created_at_utc",
        "price_proposed",
        "target_signed",
        "target_price_training_eligible",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires absentes : {sorted(missing)}")

    eligible = pd.to_numeric(raw["target_price_training_eligible"], errors="coerce") == 1
    data = raw.loc[eligible].copy()
    data["target_signed"] = pd.to_numeric(data["target_signed"], errors="coerce")
    data = data[data["target_signed"].isin([0, 1])]
    data = data[pd.to_numeric(data["price_proposed"], errors="coerce").notna()]
    data["request_created_at_utc"] = pd.to_datetime(
        data["request_created_at_utc"], errors="coerce", utc=True
    )
    data = data[data["request_created_at_utc"].notna()]
    data = data.sort_values("request_created_at_utc").reset_index(drop=True)
    data = prepare_features(data)

    class_counts = data["target_signed"].value_counts()
    minimum_rows = int(config["minimum_training_rows"])
    minimum_per_class = int(config["minimum_rows_per_class"])
    if len(data) < minimum_rows or len(class_counts) < 2 or class_counts.min() < minimum_per_class:
        raise ValueError(
            "Pas assez de données clôturées pour entraîner sérieusement le modèle. "
            f"Trouvé : {len(data)} lignes, classes : {class_counts.to_dict()}. "
            f"Minimum : {minimum_rows} lignes et {minimum_per_class} par classe."
        )

    numeric_features, categorical_features = select_feature_columns(data.columns)
    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        raise ValueError("Aucune variable autorisée n'a été trouvée.")

    split_index = int(len(data) * (1 - float(config["temporal_validation_fraction"])))
    split_index = min(max(split_index, 1), len(data) - 1)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    if train["target_signed"].nunique() < 2 or test["target_signed"].nunique() < 2:
        raise ValueError(
            "Le découpage chronologique ne contient pas les deux résultats dans chaque période. "
            "Il faut davantage d'historique ou revoir la qualité des statuts."
        )

    x_train = train[feature_columns]
    y_train = train["target_signed"].astype(int)
    x_test = test[feature_columns]
    y_test = test["target_signed"].astype(int)
    calibration_folds = max(2, min(3, int(y_train.value_counts().min())))

    seed = int(config["random_seed"])
    evaluations: dict[str, dict] = {}
    validation_models = {}
    for name in ("logistic_regression", "random_forest"):
        model = make_model(
            name,
            numeric_features,
            categorical_features,
            seed,
            calibration_folds,
        )
        model.fit(x_train, y_train)
        evaluations[name] = score_model(model, x_test, y_test)
        validation_models[name] = model

    baseline_probability = float(y_train.mean())
    baseline_vector = np.full(len(y_test), baseline_probability)
    baseline = {
        "constant_probability": baseline_probability,
        "brier_score": float(brier_score_loss(y_test, baseline_vector)),
        "log_loss": float(log_loss(y_test, baseline_vector, labels=[0, 1])),
    }

    # La calibration de probabilité est prioritaire : Brier le plus bas, puis log-loss.
    lowest_brier_name = min(
        evaluations,
        key=lambda name: (
            evaluations[name]["brier_score"],
            evaluations[name]["log_loss"],
        ),
    )
    # À qualité presque égale, la régression est plus stable pour tester des prix
    # jamais observés. La forêt ne gagne que si son Brier est >1 % meilleur.
    if (
        evaluations["logistic_regression"]["brier_score"]
        <= evaluations[lowest_brier_name]["brier_score"] * 1.01
    ):
        best_name = "logistic_regression"
    else:
        best_name = lowest_brier_name
    best_validation_model = validation_models[best_name]

    importance = permutation_importance(
        best_validation_model,
        x_test,
        y_test,
        scoring="neg_brier_score",
        n_repeats=8,
        random_state=seed,
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    final_folds = max(2, min(3, int(data["target_signed"].value_counts().min())))
    final_model = make_model(
        best_name,
        numeric_features,
        categorical_features,
        seed,
        final_folds,
    )
    final_model.fit(data[feature_columns], data["target_signed"].astype(int))

    # Le meilleur classifieur sert au score de conversion. Pour faire varier le
    # prix, un modèle linéaire séparé est plus stable entre deux prix jamais vus.
    if best_name == "logistic_regression":
        price_model = final_model
    else:
        price_model = make_model(
            "logistic_regression",
            numeric_features,
            categorical_features,
            seed,
            final_folds,
        )
        price_model.fit(data[feature_columns], data["target_signed"].astype(int))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_dir / "signature_model.joblib"
    metrics_path = args.output_dir / "training_report.json"
    importance_path = args.output_dir / "feature_importance.csv"

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "price_model": price_model,
        "price_model_name": "logistic_regression",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "training_rows": int(len(data)),
        "training_start": data["request_created_at_utc"].min().isoformat(),
        "training_end": data["request_created_at_utc"].max().isoformat(),
        "config": config,
    }
    joblib.dump(bundle, bundle_path)

    report = {
        "selected_model": best_name,
        "selection_rule": (
            "brier_score puis log_loss sur les 20 % les plus récents; "
            "préférence au modèle linéaire si son Brier est à moins de 1 % du meilleur"
        ),
        "training_rows": int(len(data)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "temporal_split_at": test["request_created_at_utc"].min().isoformat(),
        "models": evaluations,
        "naive_baseline": baseline,
        "validation_status": (
            "PASS"
            if evaluations[best_name]["brier_score"] < baseline["brier_score"]
            else "FAIL_MODEL_NOT_BETTER_THAN_BASELINE"
        ),
        "important_warning": (
            "Ces métriques mesurent une validation historique. Un test contrôlé sur les "
            "nouveaux devis reste nécessaire pour mesurer le gain commercial causal."
        ),
    }
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    importance_table.to_csv(importance_path, index=False)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nModèle enregistré : {bundle_path}")


if __name__ == "__main__":
    main()
