"""F1 Predictive Analytics Dashboard.   Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from models.simulate import title_odds
from models.train import SAVED
from pipeline.extract import PROC
from pipeline.features import FEATURES, correlation_table

st.set_page_config(page_title="F1 Predictive Analytics", layout="wide")


@st.cache_resource(show_spinner="Running pipeline (first launch only)...")
def load_everything():
    if not (SAVED / "models.joblib").exists() or not (PROC / "features.csv").exists():
        from pipeline.run_pipeline import run
        run()
    feat = pd.read_csv(PROC / "features.csv")
    pred = pd.read_csv(SAVED / "predictions.csv")
    bundle = joblib.load(SAVED / "models.joblib")
    metrics = json.loads((SAVED / "metrics.json").read_text())
    return feat, pred, bundle, metrics


feat, pred, bundle, metrics = load_everything()
test_year = bundle["test_year"]

st.title("F1 Predictive Analytics Dashboard")
st.caption(f"Data: {metrics.get('data_source', 'unknown')}  |  Models trained on seasons before "
           f"{test_year}; {test_year} is the held-out test season.")

tab_race, tab_champ, tab_insight, tab_what, tab_report = st.tabs(
    ["Next Race", "Championship", "Insights", "What-If", "Model Report"])

years = sorted(feat["year"].unique())

# ---------------------------------------------------------------- Next Race
with tab_race:
    c1, c2 = st.columns(2)
    year = c1.selectbox("Season", years, index=len(years) - 1, key="race_year")
    rounds = sorted(feat.loc[feat["year"] == year, "round"].unique())
    rnd = c2.selectbox("Round", rounds, key="race_round")
    race = pred[(pred["year"] == year) & (pred["round"] == rnd)].sort_values("win_p", ascending=False)
    if race["in_sample"].iloc[0]:
        st.warning("This season was used for training, so predictions are in-sample. "
                   f"Pick {test_year} for an honest out-of-sample view.")
    st.subheader(f"{race['circuit'].iloc[0]}  |  {year} round {rnd}")

    left, right = st.columns([3, 2])
    fig = px.bar(race.head(10), x="win_p", y="driver", orientation="h",
                 labels={"win_p": "Win probability", "driver": ""},
                 color_discrete_sequence=["#222222"])
    fig.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_xaxes(tickformat=".0%")
    left.plotly_chart(fig)

    show = race[["driver", "constructor", "grid", "win_p", "podium_p", "pred_finish", "dnf_p", "finish"]].copy()
    show.columns = ["Driver", "Team", "Grid", "Win %", "Podium %", "Pred. finish", "DNF risk %", "Actual finish"]
    for c in ("Win %", "Podium %", "DNF risk %"):
        show[c] = (show[c] * 100).round(1)
    show["Pred. finish"] = show["Pred. finish"].round(1)
    right.dataframe(show, hide_index=True, height=380)

    st.markdown("**Why does the model rate a driver this way?**")
    pick = st.selectbox("Driver", race["driver"].tolist(), key="race_driver")
    rows = feat[(feat["year"] == year) & (feat["round"] == rnd)].set_index("driver")
    x = rows.loc[[pick], FEATURES]
    try:
        import shap
        sv = shap.TreeExplainer(bundle["win"])(x)
        contrib = pd.Series(sv.values[0], index=FEATURES).sort_values()
        title = "SHAP contribution to win log-odds (right = pushes win chance up)"
    except Exception:
        contrib = (x.iloc[0] - feat[FEATURES].mean()) / feat[FEATURES].std()
        contrib = contrib.mul(pd.Series(metrics["importance"])).sort_values()
        title = "Feature importance-weighted deviation (install `shap` for exact explanations)"
    fig2 = px.bar(contrib, orientation="h", labels={"value": "", "index": ""}, title=title,
                  color_discrete_sequence=["#444444"])
    fig2.update_layout(showlegend=False, height=330, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2)

# ---------------------------------------------------------------- Championship
with tab_champ:
    c1, c2, c3 = st.columns(3)
    cy = c1.selectbox("Season", years, index=len(years) - 1, key="champ_year")
    max_r = int(feat.loc[feat["year"] == cy, "round"].max())
    after = c2.slider("Standings after round", 3, max_r - 1, max_r // 2)
    sims = c3.select_slider("Simulations", [1000, 2000, 4000, 8000], value=4000)
    odds = title_odds(feat, bundle, cy, after, n_sims=sims)
    actual = (feat[feat["year"] == cy].groupby("driver")["points"].sum()
              .sort_values(ascending=False))
    left, right = st.columns(2)
    f = px.bar(odds.head(8), x="driver", y="title_prob", color_discrete_sequence=["#222222"],
               labels={"title_prob": "Title probability", "driver": ""})
    f.update_yaxes(tickformat=".0%")
    f.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), title="Championship odds")
    left.plotly_chart(f)
    t = odds.head(10).copy()
    t["title_prob"] = (t["title_prob"] * 100).round(1)
    t["expected_final_pts"] = t["expected_final_pts"].round(0)
    t.columns = ["Driver", "Team", "Points now", "Title %", "Exp. final pts"]
    right.dataframe(t, hide_index=True, height=380)
    if len(actual):
        st.info(f"Actual {cy} champion: **{actual.index[0]}** ({int(actual.iloc[0])} pts).")
    st.caption("Monte Carlo: model-predicted strength + random race noise + DNF risk, "
               "repeated for the remaining rounds.")

# ---------------------------------------------------------------- Insights
with tab_insight:
    st.subheader("Correlation map (Spearman)")
    corr = correlation_table(feat)
    hm = px.imshow(corr.round(2), text_auto=True, aspect="auto", color_continuous_scale="Greys",
                   zmin=-1, zmax=1)
    hm.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(hm)
    c1, c2 = st.columns(2)
    box = px.box(feat, x="grid", y="finish", labels={"grid": "Grid position", "finish": "Finish"},
                 color_discrete_sequence=["#333333"], title="Grid position vs finishing position")
    c1.plotly_chart(box)
    team = (feat.groupby("constructor").agg(dnf_rate=("dnf", "mean"), avg_pts=("points", "mean"))
            .reset_index())
    sc = px.scatter(team, x="avg_pts", y="dnf_rate", text="constructor",
                    labels={"avg_pts": "Avg points per entry", "dnf_rate": "DNF rate"},
                    color_discrete_sequence=["#333333"], title="Team pace vs reliability")
    sc.update_traces(textposition="top center")
    sc.update_yaxes(tickformat=".0%")
    c2.plotly_chart(sc)

# ---------------------------------------------------------------- What-If
with tab_what:
    st.subheader("What-if simulator")
    c1, c2, c3 = st.columns(3)
    wy = c1.selectbox("Season", years, index=len(years) - 1, key="w_year")
    wr = c2.selectbox("Round", sorted(feat.loc[feat["year"] == wy, "round"].unique()), key="w_round")
    field = feat[(feat["year"] == wy) & (feat["round"] == wr)].reset_index(drop=True)
    wd = c3.selectbox("Driver", sorted(field["driver"]), key="w_driver")
    base = field[field["driver"] == wd].iloc[0]

    s1, s2, s3 = st.columns(3)
    new_grid = s1.slider("Grid position", 1, 20, int(base["grid"]))
    new_form = s2.slider("Recent avg finish (last 5)", 1.0, 20.0, float(round(base["form_pos5"], 1)), 0.5)
    new_team = s3.slider("Team recent points per race", 0.0, 60.0, float(round(base["team_pts5"], 1)), 1.0)

    def score(df):
        return (bundle["win"].predict_proba(df[FEATURES])[:, 1],
                bundle["finish"].predict(df[FEATURES]),
                bundle["dnf"].predict_proba(df[FEATURES])[:, 1])

    wp0, fp0, dp0 = score(field)
    mod = field.copy()
    i = mod.index[mod["driver"] == wd][0]
    mod.loc[i, ["grid", "form_pos5", "team_pts5"]] = [new_grid, new_form, new_team]
    wp1, fp1, dp1 = score(mod)
    win0, win1 = wp0[i] / wp0.sum(), wp1[i] / wp1.sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Win probability", f"{win1:.1%}", f"{(win1 - win0) * 100:+.1f} pts vs actual setup")
    m2.metric("Predicted finish", f"{fp1[i]:.1f}", f"{fp1[i] - fp0[i]:+.1f}", delta_color="inverse")
    m3.metric("DNF risk", f"{dp1[i]:.1%}", f"{(dp1[i] - dp0[i]) * 100:+.1f} pts", delta_color="inverse")

# ---------------------------------------------------------------- Model Report
with tab_report:
    st.subheader(f"Held-out test season: {metrics['test_season']}")
    w, p, r, d = metrics["winner"], metrics["podium"], metrics["finish_regression"], metrics["dnf"]
    table = pd.DataFrame([
        ["Race winner (top-1 accuracy)", f"{w['model_top1_acc']:.1%}", f"{w['baseline_pole_acc']:.1%}", "Baseline: pole sitter wins"],
        ["Podium (share of podium found)", f"{p['model_hit_rate']:.1%}", f"{p['baseline_grid_hit_rate']:.1%}", "Baseline: top 3 on grid"],
        ["Finish position MAE", f"{r['model_mae']:.2f}", f"{r['baseline_grid_mae']:.2f}", "Baseline: finish = grid"],
        ["Finish position RMSE", f"{r['model_rmse']:.2f}", f"{r['baseline_grid_rmse']:.2f}", "Lower is better"],
        ["Finish rank correlation", f"{r['model_spearman']:.3f}", f"{r['baseline_grid_spearman']:.3f}", "Per-race Spearman"],
        ["DNF Brier score", f"{d['model_brier']:.4f}", f"{d['baseline_brier']:.4f}", "Baseline: constant rate"],
    ], columns=["Task", "Model", "Baseline", "Note"])
    st.dataframe(table, hide_index=True)
    imp = pd.Series(metrics["importance"]).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", title="Winner model: feature importance",
                           labels={"value": "", "index": ""}, color_discrete_sequence=["#333333"])
                    .update_layout(showlegend=False, height=340))
    st.caption("Grid position dominates F1, so the honest question is how much the model "
               "adds beyond the baseline.")
