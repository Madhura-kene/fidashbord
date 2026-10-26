"""Step 3: predictive models + baseline comparison. Time-based split (last season = test)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor

from pipeline.features import FEATURES

SAVED = Path(__file__).resolve().parent / "saved"

CLF = dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", random_state=42)
REG = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=5, random_state=42)


def _top1_hit(df, score):
    return float(np.mean([g.loc[g[score].idxmax(), "win"] for _, g in df.groupby("raceId")]))


def _podium_hit(df, score):
    hits = [g.nlargest(3, score)["podium"].sum() / 3 for _, g in df.groupby("raceId")]
    return float(np.mean(hits))


def _spearman(df, score):
    return float(np.nanmean([g["finish"].corr(g[score], method="spearman")
                             for _, g in df.groupby("raceId")]))


def train_all(feat):
    SAVED.mkdir(parents=True, exist_ok=True)
    test_year = int(feat["year"].max())
    train, test = feat[feat["year"] < test_year], feat[feat["year"] == test_year].copy()
    X_tr, X_te = train[FEATURES], test[FEATURES]

    win = XGBClassifier(**CLF).fit(X_tr, train["win"])
    podium = XGBClassifier(**CLF).fit(X_tr, train["podium"])
    dnf = XGBClassifier(**CLF).fit(X_tr, train["dnf"])
    finish = XGBRegressor(**REG).fit(X_tr, train["finish"])

    test["win_p"] = win.predict_proba(X_te)[:, 1]
    test["podium_p"] = podium.predict_proba(X_te)[:, 1]
    test["dnf_p"] = dnf.predict_proba(X_te)[:, 1]
    test["pred_finish"] = finish.predict(X_te)
    test["neg_grid"] = -test["grid"]
    test["neg_pred"] = -test["pred_finish"]

    base_rate = train["dnf"].mean()
    metrics = {
        "test_season": test_year,
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "winner": {
            "model_top1_acc": _top1_hit(test, "win_p"),
            "baseline_pole_acc": _top1_hit(test, "neg_grid"),
            "log_loss": float(log_loss(test["win"], test["win_p"])),
        },
        "podium": {
            "model_hit_rate": _podium_hit(test, "podium_p"),
            "baseline_grid_hit_rate": _podium_hit(test, "neg_grid"),
        },
        "finish_regression": {
            "model_mae": float(mean_absolute_error(test["finish"], test["pred_finish"])),
            "baseline_grid_mae": float(mean_absolute_error(test["finish"], test["grid"])),
            "model_rmse": float(np.sqrt(mean_squared_error(test["finish"], test["pred_finish"]))),
            "baseline_grid_rmse": float(np.sqrt(mean_squared_error(test["finish"], test["grid"]))),
            "model_spearman": _spearman(test, "pred_finish"),
            "baseline_grid_spearman": _spearman(test, "grid"),
        },
        "dnf": {
            "model_auc": float(roc_auc_score(test["dnf"], test["dnf_p"])) if test["dnf"].nunique() > 1 else None,
            "model_brier": float(brier_score_loss(test["dnf"], test["dnf_p"])),
            "baseline_brier": float(brier_score_loss(test["dnf"], np.full(len(test), base_rate))),
        },
        "importance": dict(sorted(
            zip(FEATURES, map(float, win.feature_importances_)), key=lambda kv: -kv[1])),
    }

    joblib.dump({"win": win, "podium": podium, "dnf": dnf, "finish": finish,
                 "features": FEATURES, "test_year": test_year}, SAVED / "models.joblib")
    (SAVED / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # predictions for every row so the dashboard can browse any race
    X_all = feat[FEATURES]
    out = feat[["raceId", "year", "round", "circuit", "driver", "constructor",
                "grid", "finish", "points", "dnf"]].copy()
    out["win_p"] = win.predict_proba(X_all)[:, 1]
    out["podium_p"] = podium.predict_proba(X_all)[:, 1]
    out["dnf_p"] = dnf.predict_proba(X_all)[:, 1]
    out["pred_finish"] = finish.predict(X_all)
    out["win_p"] = out["win_p"] / out.groupby("raceId")["win_p"].transform("sum")
    out["podium_p"] = (out["podium_p"] * 3 / out.groupby("raceId")["podium_p"].transform("sum")).clip(upper=1)
    out["in_sample"] = out["year"] < test_year
    out.to_csv(SAVED / "predictions.csv", index=False)
    return metrics
