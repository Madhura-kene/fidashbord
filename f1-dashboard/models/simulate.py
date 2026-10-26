"""Championship odds via Monte Carlo over the remaining races of a season."""
import numpy as np
import pandas as pd

from pipeline.features import FEATURES

POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10)


def title_odds(feat, bundle, year, after_round, n_sims=4000, noise=3.0, seed=0):
    """Estimate each driver's title probability given results up to `after_round`.

    Strength = regression-predicted finish from each driver's latest form (grid is
    approximated by recent form since future grids are unknown). Each simulated race
    adds Gumbel noise to that strength, applies DNF risk, and awards F1 points.
    """
    rng = np.random.default_rng(seed)
    season = feat[feat["year"] == year]
    last_round = int(season["round"].max())
    done = season[season["round"] <= after_round]
    remaining = last_round - int(after_round)

    active = done[done["round"] >= max(1, after_round - 2)]["driver"].unique()
    latest = (done.sort_values("round").groupby("driver").tail(1)
              .set_index("driver").loc[lambda d: d.index.isin(active)])
    hist = done.sort_values("round").groupby("driver")
    pts = done.groupby("driver")["points"].sum().reindex(latest.index).fillna(0)

    X = latest[FEATURES].copy()
    X["form_pos5"] = hist["finish"].apply(lambda s: s.tail(5).mean()).reindex(X.index)
    X["form_pts5"] = hist["points"].apply(lambda s: s.tail(5).mean()).reindex(X.index)
    X["dnf_rate10"] = hist["dnf"].apply(lambda s: s.tail(10).mean()).reindex(X.index)
    X["season_pts_before"] = pts
    X["grid"] = X["form_pos5"].clip(1, 20)
    X["grid_vs_teammate"] = 0.0
    strength = bundle["finish"].predict(X[FEATURES])
    dnf_p = X["dnf_rate10"].to_numpy()
    n_dr = len(X)

    totals = np.tile(pts.to_numpy(), (n_sims, 1)).astype(float)
    for _ in range(remaining):
        score = strength[None, :] + rng.gumbel(0, noise, (n_sims, n_dr))
        score = np.where(rng.random((n_sims, n_dr)) < dnf_p[None, :], 1e3, score)
        rank = score.argsort(axis=1).argsort(axis=1)
        totals += POINTS[np.minimum(rank, 19)] * (score < 1e3)
    champion = totals.argmax(axis=1)
    odds = np.bincount(champion, minlength=n_dr) / n_sims
    return (pd.DataFrame({"driver": X.index, "constructor": latest.loc[X.index, "constructor"].to_numpy(),
                          "points_now": pts.to_numpy(), "title_prob": odds,
                          "expected_final_pts": totals.mean(axis=0)})
            .sort_values("title_prob", ascending=False).reset_index(drop=True))
