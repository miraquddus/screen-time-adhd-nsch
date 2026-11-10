"""Screen time and ADHD in U.S. children aged 6-17: National Survey of Children's Health 2024-2025.

Survey-weighted estimates with design-based (Taylor-linearized) standard errors, plus an
ML model that flags children likely to have 4+ hours of daily recreational screen time.

Usage:
    python src/analyze.py --data-dir data
where data/ holds nsch_2024e_topical.dta and nsch_2025e_topical.dta from the Census Bureau.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})

COLS = ["hhid", "fipsst", "stratum", "fwc", "sc_age_years", "sc_sex", "sc_race_r", "sc_hispanic_r",
        "fpl_i1", "higrade", "family_r", "screentime", "k2q31a", "k2q31b", "k2q31d", "k2q32a", "k2q32b",
        "k2q33a", "k2q33b", "hoursleep", "physactiv", "bmiclass", "ace1", "k7q30", "k7q31", "k7q32",
        "k10q11", "k10q12", "k10q40_r", "bedtime", "totkids_r", "k2q35a", "k2q30a"]
SCREEN_LABELS = {1: "<1 h", 2: "1 h", 3: "2 h", 4: "3 h", 5: "4+ h"}


# ------------------------------------------------------------------ data
def load(data_dir: Path) -> pd.DataFrame:
    frames = []
    for year in (2024, 2025):
        f = data_dir / f"nsch_{year}e_topical.dta"
        d = pd.read_stata(f, columns=COLS, convert_categoricals=False)
        d["year"] = year
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.sc_age_years >= 6) & (df.sc_age_years <= 17) & df.screentime.between(1, 5)].copy()

    # pooled two-year weights: divide by number of years (Census multi-year guidance)
    df["w"] = df.fwc / 2
    df["strata"] = df.fipsst.astype(str) + "_" + df.stratum.astype(str) + "_" + df.year.astype(str)

    def current(ever, now):
        out = pd.Series(np.nan, index=df.index)
        out[df[ever] == 2] = 0
        out[(df[ever] == 1) & (df[now] == 1)] = 1
        out[(df[ever] == 1) & (df[now] == 2)] = 0
        return out

    df["adhd"] = current("k2q31a", "k2q31b")
    df["anxiety"] = current("k2q33a", "k2q33b")
    df["depression"] = current("k2q32a", "k2q32b")
    # AASM sleep recommendation: 9+ h for 6-12, 8+ h for 13-17 (HOURSLEEP 4 = 8 h, 5 = 9 h)
    need = np.where(df.sc_age_years <= 12, 5, 4)
    df["short_sleep"] = np.where(df.hoursleep.between(1, 7), (df.hoursleep < need).astype(float), np.nan)
    df["inactive"] = np.where(df.physactiv.between(1, 4), (df.physactiv == 1).astype(float), np.nan)
    df["obesity"] = np.where(df.bmiclass.between(1, 4), (df.bmiclass == 4).astype(float), np.nan)

    df["screen"] = df.screentime.astype(int)
    df["screen4"] = (df.screen == 5).astype(int)
    df["age_group"] = pd.cut(df.sc_age_years, [5, 11, 14, 17], labels=["6-11", "12-14", "15-17"])
    df["sex"] = df.sc_sex.map({1: "Male", 2: "Female"})
    race = df.sc_race_r.map({1: "White", 2: "Black", 3: "Other", 4: "Asian", 5: "Other", 6: "Other", 7: "Multiracial"})
    df["race_eth"] = np.where(df.sc_hispanic_r == 1, "Hispanic", "NH " + race.fillna("Other"))
    df["poverty"] = pd.cut(df.fpl_i1, [0, 99, 199, 399, 1000], labels=["<100% FPL", "100-199%", "200-399%", "400%+"])
    df["parent_edu"] = df.higrade.map({1: "<HS", 2: "HS", 3: ">HS"})
    df["family"] = df.family_r.map({1: "Two parents, married", 2: "Two parents, unmarried", 3: "Two parents, married",
                                    4: "Two parents, unmarried", 5: "Single mother", 6: "Single father",
                                    7: "Other", 8: "Other"})
    return df


# ------------------------------------------------------------------ survey statistics
def design_cov(scores: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Stratified with-replacement variance of summed scores (one child per household = PSU)."""
    k = scores.shape[1]
    V = np.zeros((k, k))
    for h in np.unique(strata):
        s = scores[strata == h]
        n = len(s)
        if n < 2:
            continue
        d = s - s.mean(axis=0)
        V += n / (n - 1) * d.T @ d
    return V


def weighted_prop(y, w, strata):
    m = ~np.isnan(y)
    y, w, strata = y[m], w[m], strata[m]
    p = np.sum(w * y) / np.sum(w)
    z = (w * (y - p) / np.sum(w))[:, None]
    se = np.sqrt(design_cov(z, strata)[0, 0])
    return p, se, int(m.sum())


def svy_logit(y, X, w, strata, names):
    """Weighted logistic regression with Taylor-linearized design-based SEs."""
    m = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y, X, w, strata = y[m], X[m], w[m], strata[m]
    w = w / w.mean()

    def nll(b):
        eta = X @ b
        return -np.sum(w * (y * eta - np.logaddexp(0, eta)))

    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        return -X.T @ (w * (y - p))

    b = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS",
                          options={"gtol": 1e-7, "maxiter": 2000}).x
    p = 1 / (1 + np.exp(-(X @ b)))
    H = (X * (w * p * (1 - p))[:, None]).T @ X
    U = X * (w * (y - p))[:, None]
    Hinv = np.linalg.inv(H)
    V = Hinv @ design_cov(U, strata) @ Hinv
    se = np.sqrt(np.diag(V))
    out = pd.DataFrame({"term": names, "or": np.exp(b), "lo": np.exp(b - 1.96 * se),
                        "hi": np.exp(b + 1.96 * se), "p": 2 * stats.norm.sf(np.abs(b / se))})
    return out, int(m.sum())


def design_matrix(df, cols, ref):
    parts = [pd.Series(1.0, index=df.index, name="Intercept")]
    for c in cols:
        dummies = pd.get_dummies(df[c].astype("object"), prefix=c, dtype=float)
        dummies = dummies.drop(columns=f"{c}_{ref[c]}")
        parts.append(dummies)
    X = pd.concat(parts, axis=1)
    X[df[cols].isna().any(axis=1)] = np.nan
    return X


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args()
    df = load(Path(args.data_dir))
    st = df.strata.values
    w = df.w.values
    metrics: dict = {"n_children_6_17": int(len(df))}

    # 1. distribution of screen time by age group
    dist = []
    for ag in ["6-11", "12-14", "15-17"]:
        sub = df[df.age_group == ag]
        for k, lab in SCREEN_LABELS.items():
            p, se, n = weighted_prop((sub.screen == k).astype(float).values, sub.w.values, sub.strata.values)
            dist.append({"age_group": ag, "screen": lab, "pct": 100 * p, "se": 100 * se})
    dist = pd.DataFrame(dist)
    metrics["screen_distribution"] = dist.round(2).to_dict(orient="records")
    p4, se4, _ = weighted_prop(df.screen4.astype(float).values, w, st)
    metrics["pct_4plus_hours_all"] = [round(100 * p4, 1), round(100 * 1.96 * se4, 1)]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = ["#bcd5f3", "#8ab6ea", "#5a96e0", BLUE, "#1c4f8f"]   # one hue, light -> dark
    left = np.zeros(3)
    for k, lab in enumerate(SCREEN_LABELS.values()):
        vals = dist[dist.screen == lab].set_index("age_group").loc[["6-11", "12-14", "15-17"], "pct"].values
        ax.barh(["6-11", "12-14", "15-17"], vals, left=left, color=colors[k], label=lab, height=0.6,
                edgecolor="#fcfcfb", linewidth=2)
        for i, v in enumerate(vals):
            if v > 7:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        color="white" if k >= 2 else INK, fontsize=9)
        left += vals
    ax.set_xlim(0, 100); ax.set_xlabel("% of children (weighted)"); ax.set_ylabel("Age")
    ax.set_title("Weekday recreational screen time by age", loc="left", pad=26)
    ax.legend(frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "01_screen_time_by_age.png", dpi=160); plt.close(fig)

    # 2. ADHD prevalence by screen time (dose-response, crude)
    prev = []
    for k, lab in SCREEN_LABELS.items():
        sub = df[df.screen == k]
        p, se, n = weighted_prop(sub.adhd.values, sub.w.values, sub.strata.values)
        prev.append({"screen": lab, "pct": 100 * p, "lo": 100 * (p - 1.96 * se), "hi": 100 * (p + 1.96 * se), "n": n})
    prev = pd.DataFrame(prev)
    metrics["adhd_prevalence_by_screen"] = prev.round(2).to_dict(orient="records")
    pa, sea, _ = weighted_prop(df.adhd.values, w, st)
    metrics["adhd_prevalence_overall"] = [round(100 * pa, 1), round(100 * 1.96 * sea, 1)]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.errorbar(prev.screen, prev.pct, yerr=[prev.pct - prev.lo, prev.hi - prev.pct], color=BLUE, lw=2,
                marker="o", ms=6, capsize=4)
    for i, r in prev.iterrows():
        ax.text(i, r.hi + 0.6, f"{r.pct:.1f}%", ha="center", color=INK2, fontsize=9)
    ax.set_ylim(0, prev.hi.max() + 4)
    ax.set_xlabel("Weekday recreational screen time"); ax.set_ylabel("% with current ADHD (95% CI)")
    ax.set_title("Current ADHD by screen time, ages 6-17 (unadjusted)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "02_adhd_by_screen_time.png", dpi=160); plt.close(fig)

    # 3. adjusted odds ratios for several outcomes
    covs = ["screen_cat", "age_group", "sex", "race_eth", "poverty", "parent_edu", "family", "year"]
    ref = {"screen_cat": "<1 h", "age_group": "6-11", "sex": "Female", "race_eth": "NH White",
           "poverty": "400%+", "parent_edu": ">HS", "family": "Two parents, married", "year": 2024}
    df["screen_cat"] = df.screen.map(SCREEN_LABELS)
    X = design_matrix(df, covs, ref)
    outcomes = {"adhd": "Current ADHD", "anxiety": "Current anxiety", "depression": "Current depression",
                "short_sleep": "Short sleep (below AASM)", "inactive": "No days of 60-min activity"}
    or_rows, adhd_table = [], None
    for o, label in outcomes.items():
        tab, n = svy_logit(df[o].values.astype(float), X.values, w, st, list(X.columns))
        if o == "adhd":
            adhd_table = tab
            metrics["adhd_model_n"] = n
        for lab in ["1 h", "2 h", "3 h", "4+ h"]:
            r = tab[tab.term == f"screen_cat_{lab}"].iloc[0]
            or_rows.append({"outcome": label, "screen": lab, "or": r["or"], "lo": r.lo, "hi": r.hi, "p": r.p, "n": n})
    ors = pd.DataFrame(or_rows)
    ors.round(4).to_csv(ROOT / "reports" / "adjusted_odds_ratios.csv", index=False)
    adhd_table.round(4).to_csv(ROOT / "reports" / "adhd_full_model.csv", index=False)
    metrics["adjusted_or_4plus_vs_lt1"] = ors[ors.screen == "4+ h"].round(3).to_dict(orient="records")

    # sensitivity: also adjust for sleep and physical activity (possible mediators)
    df["sleep_cat"] = df.short_sleep.map({0.0: "Adequate", 1.0: "Short"})
    df["activity_cat"] = df.physactiv.map({1: "0 days", 2: "1-3 days", 3: "4-6 days", 4: "Daily"})
    X2 = design_matrix(df, covs + ["sleep_cat", "activity_cat"], {**ref, "sleep_cat": "Adequate", "activity_cat": "Daily"})
    tab2, n2 = svy_logit(df.adhd.values.astype(float), X2.values, w, st, list(X2.columns))
    r2 = tab2[tab2.term == "screen_cat_4+ h"].iloc[0]
    metrics["adhd_or_4plus_adjusted_plus_sleep_activity"] = {"or": round(r2["or"], 3), "lo": round(r2.lo, 3), "hi": round(r2.hi, 3), "n": n2}

    fig, ax = plt.subplots(figsize=(7, 3.6))
    o4 = ors[ors.screen == "4+ h"].iloc[::-1].reset_index(drop=True)
    ax.errorbar(o4["or"], o4.outcome, xerr=[o4["or"] - o4.lo, o4.hi - o4["or"]], fmt="o", color=BLUE,
                ms=7, capsize=4, lw=2)
    for i, r in o4.iterrows():
        ax.text(r.hi + 0.06, i, f"{r['or']:.2f} ({r.lo:.2f}-{r.hi:.2f})", va="center", color=INK2, fontsize=9)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_xlim(0.5, max(o4.hi) + 0.9)
    ax.set_xlabel("Adjusted odds ratio, 4+ h vs <1 h per weekday (95% CI)")
    ax.set_title("Heavy screen time and child health outcomes", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "03_adjusted_odds_ratios.png", dpi=160); plt.close(fig)

    # 4. ML: who is at risk of 4+ h screen time? (train 2024, test 2025)
    feats = ["sc_age_years", "sex", "race_eth", "poverty", "parent_edu", "family", "totkids_r", "physactiv",
             "hoursleep", "bedtime", "k7q30", "k7q31", "k7q32", "k10q11", "k10q12", "k10q40_r", "ace1"]
    D = df[feats + ["screen4", "year", "w"]].copy()
    cat = ["sex", "race_eth", "poverty", "parent_edu", "family"]
    for c in cat:
        D[c] = D[c].astype("object").fillna("Missing").astype("category")
    for c in [f for f in feats if f not in cat]:
        D[c] = pd.to_numeric(D[c], errors="coerce").where(lambda s: s < 90)  # drop NSCH reserve codes
    tr, te = D.year == 2024, D.year == 2025
    gb = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=400, learning_rate=0.05,
                                        max_leaf_nodes=31, random_state=0)
    gb.fit(D.loc[tr, feats], D.loc[tr, "screen4"], sample_weight=D.loc[tr, "w"] / D.loc[tr, "w"].mean())
    p_gb = gb.predict_proba(D.loc[te, feats])[:, 1]
    num = [f for f in feats if f not in cat]
    Dl = D.copy()
    for c in num:
        Dl[c] = Dl[c].fillna(Dl.loc[tr, c].median()).astype(float)
    Dl_enc = pd.get_dummies(Dl[feats], columns=cat, dtype=float)
    lr = make_pipeline(LogisticRegression(max_iter=5000, C=0.5))
    from sklearn.preprocessing import StandardScaler
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))
    lr.fit(Dl_enc[tr], D.loc[tr, "screen4"], logisticregression__sample_weight=D.loc[tr, "w"] / D.loc[tr, "w"].mean())
    p_lr = lr.predict_proba(Dl_enc[te])[:, 1]
    yte, wte = D.loc[te, "screen4"], D.loc[te, "w"]
    age_only = D.loc[tr].groupby("sc_age_years").apply(lambda g: np.average(g.screen4, weights=g.w))
    p_age = D.loc[te, "sc_age_years"].map(age_only)
    metrics["ml_screen4"] = {
        "train_n_2024": int(tr.sum()), "test_n_2025": int(te.sum()),
        "auc_gradient_boosting": round(roc_auc_score(yte, p_gb, sample_weight=wte), 3),
        "auc_logistic": round(roc_auc_score(yte, p_lr, sample_weight=wte), 3),
        "auc_age_only_baseline": round(roc_auc_score(yte, p_age, sample_weight=wte), 3),
    }
    pi = permutation_importance(gb, D.loc[te, feats], yte, scoring="roc_auc", n_repeats=10, random_state=0)
    names = {"sc_age_years": "Age", "hoursleep": "Weeknight sleep hours", "physactiv": "Days of physical activity",
             "bedtime": "Consistent bedtime", "poverty": "Family poverty level", "parent_edu": "Parent education",
             "race_eth": "Race/ethnicity", "family": "Family structure", "k7q30": "Sports team", "k7q31": "Clubs",
             "k7q32": "Organized lessons", "totkids_r": "Children in household", "sex": "Sex",
             "k10q11": "Sidewalks nearby", "k10q12": "Park nearby", "k10q40_r": "Safe neighborhood",
             "ace1": "Hard to afford basics"}
    imp = pd.Series(pi.importances_mean, index=[names[f] for f in feats]).sort_values()
    metrics["ml_screen4"]["permutation_importance"] = imp.sort_values(ascending=False).round(4).to_dict()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(imp.index, imp.values, color=AQUA, height=0.6)
    ax.set_xlabel("Drop in ROC AUC when shuffled (2025 test set)")
    ax.set_title("What predicts 4+ hours of daily screen time", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "04_screen4_drivers.png", dpi=160); plt.close(fig)

    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
