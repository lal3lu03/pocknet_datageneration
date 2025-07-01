import argparse
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import wandb
from pathlib import Path

from model_utils import stratified_split, compute_metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_feature_order(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(cfg_path):
    cfg = load_config(cfg_path)
    base = Path(cfg_path).parent

    feature_path = (base / cfg["features_file"]).resolve()
    data_path = (base / cfg["train_csv"]).resolve()
    test_path = (base / cfg.get("test_csv", "")).resolve() if cfg.get("test_csv") else None

    feature_order = load_feature_order(feature_path)

    data = pd.read_csv(data_path)
    X = data[feature_order]
    y = data["class"].astype(int)

    test_data = None
    if test_path and test_path.exists():
        test_df = pd.read_csv(test_path)
        test_data = (test_df[feature_order], test_df["class"].astype(int))

    X_train, X_val, y_train, y_val = stratified_split(
        X, y, test_size=0.2, random_state=cfg.get("seed", 42)
    )

    rf_params = {
        "n_estimators": cfg.get("rf_trees", 100),
        "max_depth": None if cfg.get("rf_depth", 0) == 0 else cfg.get("rf_depth"),
        "max_features": "sqrt" if cfg.get("rf_features", 0) == 0 else cfg.get("rf_features"),
        "n_jobs": -1 if cfg.get("rf_threads", 0) == 0 else cfg.get("rf_threads"),
        "random_state": cfg.get("seed", 42),
        "bootstrap": True,
    }
    if cfg.get("rf_bagsize", 100) != 100:
        rf_params["max_samples"] = cfg["rf_bagsize"] / 100.0

    wandb.init(project="p2rank_rf", config=rf_params, reinit=True, mode="offline")

    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, y_pred, y_prob)

    wandb.log({k: v for k, v in metrics.items() if k != "confusion_matrix"})
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_val.tolist(),
        preds=y_pred.tolist(),
        class_names=["0", "1"],
    )})
    print("Validation Metrics:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"{k}:\n{v}")
        else:
            print(f"{k}: {v}")

    print("Classification Report:\n", classification_report(y_val, y_pred))

    if test_data is not None:
        X_test, y_test = test_data
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_pred, test_prob)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"})
        wandb.log({"test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test.tolist(),
            preds=test_pred.tolist(),
            class_names=["0", "1"],
        )})
        print("Test Metrics:")
        for k, v in test_metrics.items():
            if k == "confusion_matrix":
                print(f"{k}:\n{v}")
            else:
                print(f"{k}: {v}")
        print("Test Classification Report:\n", classification_report(y_test, test_pred))

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest like P2Rank")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)

