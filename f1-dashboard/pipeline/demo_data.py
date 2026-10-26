"""Synthetic F1-style results so the whole project runs without any download.

This is DEMO data (clearly flagged in the dashboard). Drop the real Kaggle
Ergast CSVs into data/raw/ and the pipeline switches to them automatically.
"""
import numpy as np
import pandas as pd

TEAMS = ["Aurora", "Vertex", "Helix", "Nimbus", "Quasar",
         "Ember", "Zephyr", "Orion", "Pulse", "Talon"]

DRIVERS = [
    "Alex Rowan", "Bruno Keller", "Carlos Medina", "Dev Malhotra", "Elias Brandt",
    "Farid Haddad", "Gabe Lindqvist", "Hiro Tanabe", "Ivan Petrov", "Jonas Weber",
    "Kai Nakamura", "Luca Ferrante", "Mateo Silva", "Nikhil Rao", "Oscar Dahl",
    "Pablo Reyes", "Quinn Foster", "Rohan Mehta", "Sven Larsen", "Tomas Novak",
    "Uri Cohen", "Viktor Hale", "Will Okafor", "Xavi Torres", "Yusuf Demir",
    "Zane Calloway",
]

CIRCUITS = [
    "Melbourne", "Sakhir", "Shanghai", "Baku", "Miami", "Imola", "Monaco",
    "Barcelona", "Montreal", "Spielberg", "Silverstone", "Budapest", "Spa",
    "Zandvoort", "Monza", "Singapore", "Suzuka", "Austin", "Mexico City",
    "Sao Paulo", "Las Vegas", "Lusail", "Abu Dhabi",
]

POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def generate(start=2014, end=2025, seed=7):
    rng = np.random.default_rng(seed)
    skill = {d: rng.normal(0, 1) for d in DRIVERS}
    affinity = {(d, c): rng.normal(0, 0.35) for d in DRIVERS for c in CIRCUITS}
    team_str = {t: rng.normal(0, 1.3) for t in TEAMS}
    team_rel = {t: rng.uniform(0.03, 0.12) for t in TEAMS}

    rows, race_id = [], 0
    roster = list(DRIVERS)
    rng.shuffle(roster)
    for year in range(start, end + 1):
        for t in TEAMS:
            team_str[t] += rng.normal(0, 0.45)
        if year > start:  # lineups stay mostly stable, like real F1
            for _ in range(2):  # two drivers swap seats
                i, j = rng.choice(20, size=2, replace=False)
                roster[i], roster[j] = roster[j], roster[i]
            for _ in range(2):  # two drivers replaced from the bench
                i, j = int(rng.integers(0, 20)), int(rng.integers(20, len(roster)))
                roster[i], roster[j] = roster[j], roster[i]
        lineup = {d: TEAMS[i // 2] for i, d in enumerate(roster[:20])}
        calendar = list(rng.choice(CIRCUITS, size=22, replace=False))
        for rnd, circuit in enumerate(calendar, start=1):
            race_id += 1
            date = pd.Timestamp(year, 3, 1) + pd.Timedelta(days=7 * rnd)
            drivers = list(lineup)
            car = np.array([team_str[lineup[d]] for d in drivers])
            drv = np.array([skill[d] for d in drivers])
            aff = np.array([affinity[(d, circuit)] for d in drivers])
            quali = car + 0.6 * drv + aff + rng.normal(0, 0.55, 20)
            grid = (-quali).argsort().argsort() + 1
            race = 0.55 * quali + 0.45 * (car + 0.6 * drv) + rng.normal(0, 0.8, 20)
            dnf = rng.random(20) < np.array([team_rel[lineup[d]] for d in drivers])
            order_score = np.where(dnf, -1e3 + rng.normal(0, 1, 20), race)
            finish = (-order_score).argsort().argsort() + 1
            for i, d in enumerate(drivers):
                rows.append({
                    "raceId": race_id, "year": year, "round": rnd, "date": date,
                    "circuit": circuit, "driver": d, "constructor": lineup[d],
                    "grid": int(grid[i]), "finish": int(finish[i]),
                    "points": POINTS[finish[i] - 1] if finish[i] <= 10 and not dnf[i] else 0,
                    "status": "Retired" if dnf[i] else "Finished",
                    "dnf": int(dnf[i]),
                })
    return pd.DataFrame(rows)
