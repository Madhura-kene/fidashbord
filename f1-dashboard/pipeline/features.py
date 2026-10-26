"""Step 2: feature engineering. Every feature uses only information known BEFORE the race."""
import numpy as np
import pandas as pd

FEATURES = [
    "grid", "grid_vs_teammate", "form_pos5", "form_pts5", "dnf_rate10",
    "team_pts5", "circuit_avg_pos", "season_pts_before", "champ_pos_before",
]


def _rolling_prior(df, by, col, window, min_periods=1):
    return df.groupby(by)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())


def build_features(df):
    df = df.sort_values(["year", "round", "finish"]).reset_index(drop=True).copy()
    df["race_idx"] = df.groupby(["year", "round"]).ngroup()
    df["grid"] = df["grid"].replace(0, 20)  # pit-lane start
    df["win"] = (df["finish"] == 1).astype(int)
    df["podium"] = (df["finish"] <= 3).astype(int)

    # driver form
    df["form_pos5"] = _rolling_prior(df, "driver", "finish", 5)
    df["form_pts5"] = _rolling_prior(df, "driver", "points", 5)
    df["dnf_rate10"] = _rolling_prior(df, "driver", "dnf", 10)

    # constructor form: points the team scored per race, averaged over its last 5 races
    team_race = (df.groupby(["constructor", "race_idx"], as_index=False)["points"].sum()
                 .sort_values(["constructor", "race_idx"]))
    team_race["team_pts5"] = _rolling_prior(team_race, "constructor", "points", 5)
    df = df.merge(team_race[["constructor", "race_idx", "team_pts5"]],
                  on=["constructor", "race_idx"], how="left")

    # grid relative to teammate
    team_mean = df.groupby(["race_idx", "constructor"])["grid"].transform("mean")
    df["grid_vs_teammate"] = df["grid"] - team_mean

    # driver history at this circuit (prior visits only)
    df["circuit_avg_pos"] = df.groupby(["driver", "circuit"])["finish"].transform(
        lambda s: s.shift(1).expanding().mean())

    # championship situation going into the race
    df["season_pts_before"] = (df.groupby(["driver", "year"])["points"].cumsum() - df["points"])
    df["champ_pos_before"] = df.groupby("race_idx")["season_pts_before"].rank(
        ascending=False, method="min")

    # fill gaps for debut / first-visit rows
    df["form_pos5"] = df["form_pos5"].fillna(10.5)
    df["form_pts5"] = df["form_pts5"].fillna(0)
    df["dnf_rate10"] = df["dnf_rate10"].fillna(df["dnf"].mean())
    df["team_pts5"] = df["team_pts5"].fillna(0)
    df["circuit_avg_pos"] = df["circuit_avg_pos"].fillna(df["form_pos5"])
    return df


def correlation_table(feat):
    cols = FEATURES + ["finish", "win", "dnf"]
    return feat[cols].corr(method="spearman")
