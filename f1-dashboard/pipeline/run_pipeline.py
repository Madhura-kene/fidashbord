"""Run the full pipeline:  python -m pipeline.run_pipeline"""
import json

from models.train import SAVED, train_all
from pipeline.extract import PROC, load_results
from pipeline.features import build_features


def run():
    PROC.mkdir(parents=True, exist_ok=True)
    df, source = load_results()
    df.to_csv(PROC / "results_clean.csv", index=False)
    feat = build_features(df)
    feat.to_csv(PROC / "features.csv", index=False)
    metrics = train_all(feat)
    metrics["data_source"] = source
    (SAVED / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[ok] {len(df):,} rows | source: {source}")
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k != "importance"}, indent=2))
