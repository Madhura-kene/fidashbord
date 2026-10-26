"""Step 1: extraction. Loads the Kaggle Ergast CSVs if present, else demo data."""
from pathlib import Path

import pandas as pd

from pipeline.demo_data import generate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

REQUIRED = ["results.csv", "races.csv", "drivers.csv", "constructors.csv", "status.csv"]


def has_real_data(raw=RAW):
    return all((raw / f).exists() for f in REQUIRED)


def load_ergast(raw=RAW, min_year=2010):
    """Kaggle 'Formula 1 World Championship (1950-2020)' schema."""
    read = lambda f: pd.read_csv(raw / f, na_values=["\\N"])
    results, races = read("results.csv"), read("races.csv")
    drivers, cons, status = read("drivers.csv"), read("constructors.csv"), read("status.csv")

    df = (results
          .merge(races[["raceId", "year", "round", "circuitId", "date"]], on="raceId")
          .merge(drivers[["driverId", "forename", "surname"]], on="driverId")
          .merge(cons[["constructorId", "name"]], on="constructorId")
          .merge(status, on="statusId"))
    df["driver"] = df["forename"] + " " + df["surname"]
    df = df.rename(columns={"name": "constructor", "positionOrder": "finish"})
    df["circuit"] = df["circuitId"].astype(str)
    finished = df["status"].eq("Finished") | df["status"].str.startswith("+")
    df["dnf"] = (~finished).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    cols = ["raceId", "year", "round", "date", "circuit", "driver", "constructor",
            "grid", "finish", "points", "status", "dnf"]
    return df.loc[df["year"] >= min_year, cols].reset_index(drop=True)


def load_results():
    """Returns (dataframe, source_label)."""
    if has_real_data():
        return load_ergast(), "Kaggle Ergast data"
    return generate(), "DEMO data (synthetic) - add Kaggle CSVs to data/raw/ for real results"
