# F1 Predictive Analytics Dashboard

Pipeline: extraction -> feature engineering -> correlation mapping -> models -> Streamlit dashboard.

## Run
```
pip install -r requirements.txt
python -m pipeline.run_pipeline      # builds features + trains models
streamlit run app.py
```
(The app also runs the pipeline automatically on first launch.)

## Data
- Out of the box it uses **synthetic demo data** so everything runs offline.
- For real results, download the Kaggle dataset "Formula 1 World Championship (1950-2020)" and put
  `results.csv, races.csv, drivers.csv, constructors.csv, status.csv` in `data/raw/`.
  The pipeline detects them and switches automatically (seasons from 2010 onward).

## Models (time-based split, last season is the test set)
| Task | Model | Baseline |
|---|---|---|
| Race winner | XGBoost classifier | pole sitter wins |
| Podium | XGBoost classifier | top 3 on grid |
| Finishing position | XGBoost regressor | finish = grid |
| DNF risk | XGBoost classifier | constant rate |
| Championship odds | Monte Carlo over remaining races | - |

## Layout
```
pipeline/   demo_data.py, extract.py, features.py, run_pipeline.py
models/     train.py, simulate.py, saved/ (models + metrics + predictions)
app.py      Streamlit dashboard (Next Race, Championship, Insights, What-If, Model Report)
```
