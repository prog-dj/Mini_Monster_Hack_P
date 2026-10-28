"""
MINI MONSTER HACK - PREDICT STAGE
==================================
Builds an early-warning classifier: "which students show early signs of
becoming a future BEHAVIOURAL monster (Ghost / Zombie / Vampire / Werewolf)?"

STRICT NO-LEAKAGE DESIGN
-------------------------
1. LABELS come from the TRAINING data's FULL academic year (we already know
   who became a monster there - that's the whole point of HUNT/TEST).
2. FEATURES for EVERY student (training AND predict) are computed using
   ONLY rows with DATE <= CUTOFF_DATE (2023-11-14). The model never sees
   anything that happened after the cutoff when learning patterns, and the
   predict-set naturally only contains pre-cutoff data anyway.
3. Train/test split is on STUDENTS (not rows), stratified by label, and all
   preprocessing (impute, scale) lives inside an sklearn Pipeline fitted
   ONLY on the training fold - exactly as described in the attached
   Logistic Regression / Naive Bayes / KNN / SVM guides.
4. The model is evaluated once on a held-out split of the TRAINING data
   (where we know the true outcome), THEN retrained on all training data,
   THEN applied to the separate PREDICT folder (real unknowns - no labels
   exist there, by definition of the task).

FOLDER LAYOUT (three separate folders, per your setup)
--------------------------------------------------------
  TRAIN_DIR   -> env var MMH_TRAIN_DIR   (the 80% training la_mmh_*.csv files)
  PREDICT_DIR -> env var MMH_PREDICT_DIR (the "predict" folder la_mmh_*.csv files,
                                           data available only up to 14 Nov 2023)
  OUT_DIR     -> env var MMH_PREDICT_OUT (defaults to PREDICT_DIR)

Run:
    $env:MMH_TRAIN_DIR   = "C:\\...\\mini monster hack"      # has combined_* or raw la_mmh_* files
    $env:MMH_PREDICT_DIR = "C:\\Users\\alber\\Desktop\\predict"
    python predict.py

Outputs (written to OUT_DIR):
    predict_features.csv     -> engineered features per student in the predict set
    predict_results.csv      -> P_ID, risk_score, tuned risk threshold, predicted flag
    model_comparison.csv     -> cross-validated metrics for every model tried
    heldout_metrics.csv      -> untouched holdout metrics for the selected model
    model_comparison.png     -> cross-validated model metric charts
    heldout_diagnostics.png  -> holdout precision-recall, ROC, and confusion matrix
    predictor_dashboard.png  -> four-panel early-warning/verification summary
    training_labels.csv      -> P_ID, is_future_monster (audit trail, training data only)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, make_scorer,
)

# ==================================================================
# CONFIG
# ==================================================================
TRAIN_DIR   = os.environ.get("MMH_TRAIN_DIR", ".")
PREDICT_DIR = os.environ.get("MMH_PREDICT_DIR", "./predict")
OUT_DIR     = os.environ.get("MMH_PREDICT_OUT", PREDICT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2023-11-14")  # data available only up to this date
RANDOM_STATE = 42

FILE_KEYWORDS = {
    "calendar":     ["calendar"],
    "person":       ["person_dim"],
    "module_inst":  ["module_instance_dim"],
    "pmi_fact":     ["person_module_instance_fact"],
    "video_dim":    ["video_dim"],
    "video_fact":   ["video_activity", "fact"],
    "lms_dim":      ["lms_item_dim"],
    "lms_fact":     ["lms_item_activity", "fact"],
}
OPTIONAL_FILES = {"calendar", "module_inst", "lms_dim"}
USECOLS = {
    "lms_fact":   ["P_ID", "DATE", "TIMESTAMP", "LMS_ITEM_ID", "INTERACTION_COUNT"],
    "video_fact": ["P_ID", "DATE", "TIMESTAMP", "VIDEO_ID", "TOTAL_MINUTES_DELIVERED"],
}
DTYPES = {
    "lms_fact":   {"P_ID": "category", "LMS_ITEM_ID": "category", "INTERACTION_COUNT": "float32"},
    "video_fact": {"P_ID": "category", "VIDEO_ID": "category", "TOTAL_MINUTES_DELIVERED": "float32"},
}
PARSE_DATES = {"lms_fact": ["DATE", "TIMESTAMP"], "video_fact": ["DATE", "TIMESTAMP"]}
NA_VALUES = ["null", "NULL", "", "NaN", "nan"]

COL_PID, COL_DATE, COL_TS = "P_ID", "DATE", "TIMESTAMP"

AUTUMN_START = pd.Timestamp("2023-10-02")
AUTUMN_END   = pd.Timestamp("2024-01-06")
SPRING_START = pd.Timestamp("2024-01-07")
SPRING_END   = pd.Timestamp("2024-04-27")


# ==================================================================
# STEP 1 - GENERIC LOADER (same fuzzy-match logic as pd.py)
# ==================================================================
def _find_csv(data_dir, keywords):
    all_csvs = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    matches = [f for f in all_csvs if all(kw.lower() in f.lower() for kw in keywords)]
    return matches, all_csvs


def load(data_dir, name, required=True):
    keywords = FILE_KEYWORDS[name]
    matches, all_csvs = _find_csv(data_dir, keywords)
    if not matches:
        if not required:
            print(f"[skip] no file for optional '{name}' in {data_dir}")
            return None
        raise FileNotFoundError(
            f"Could not find a CSV matching {keywords} for '{name}' in '{os.path.abspath(data_dir)}'.\n"
            f"Files present: {all_csvs}")
    if len(matches) > 1:
        print(f"[warn] multiple matches for {keywords}: {matches} -> using {matches[0]}")
    path = os.path.join(data_dir, matches[0])
    kwargs = dict(na_values=NA_VALUES)
    if name in USECOLS:
        kwargs["usecols"] = lambda c: c in USECOLS[name]
    if name in DTYPES:
        kwargs["dtype"] = DTYPES[name]
    if name in PARSE_DATES:
        kwargs["parse_dates"] = PARSE_DATES[name]
    print(f"[loading] ({data_dir}) {matches[0]} ...")
    df = pd.read_csv(path, **kwargs)
    print(f"[loaded]  {matches[0]:50s} shape={df.shape}")
    return df


def _to_naive_datetime(s):
    s = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s


def build_activity(data_dir):
    """Load + merge one folder's raw CSVs into a single long activity table."""
    person     = load(data_dir, "person")
    video_dim  = load(data_dir, "video_dim")
    video_fact = load(data_dir, "video_fact")
    lms_fact   = load(data_dir, "lms_fact")
    pmi_fact   = load(data_dir, "pmi_fact", required=False)

    for c in ["registration_date", "deregistration_date"]:
        if c in person.columns:
            person[c] = _to_naive_datetime(person[c])
    for df_ in (video_fact, lms_fact):
        for c in ["DATE", "TIMESTAMP"]:
            if c in df_.columns:
                df_[c] = _to_naive_datetime(df_[c])

    lms = lms_fact.copy()
    lms["source"] = "LMS"
    lms["item_id"] = lms["LMS_ITEM_ID"]
    lms["activity_value"] = lms["INTERACTION_COUNT"]

    vid = video_fact.merge(video_dim, on="VIDEO_ID", how="left")
    vid["source"] = "VIDEO"
    vid["item_id"] = vid["VIDEO_ID"]
    vid["activity_value"] = vid["TOTAL_MINUTES_DELIVERED"]

    cols = ["P_ID", "DATE", "TIMESTAMP", "source", "item_id", "activity_value"]
    activity = pd.concat([lms.reindex(columns=cols), vid.reindex(columns=cols)],
                          ignore_index=True)
    activity = activity.merge(person, on="P_ID", how="left")
    activity = activity.dropna(subset=[COL_TS, COL_PID]).sort_values([COL_PID, COL_TS])
    print(f"[built] activity table for '{data_dir}' shape={activity.shape}")
    return activity, person, pmi_fact


# ==================================================================
# STEP 2 - FEATURE ENGINEERING (identical function used for BOTH
# training and predict data - this is what prevents leakage/asymmetry)
# ==================================================================
def engineer_features(activity: pd.DataFrame, cutoff=CUTOFF_DATE) -> pd.DataFrame:
    """
    Build one row of early-warning features per student, using ONLY
    activity with DATE <= cutoff. Every feature here must be computable
    in real time on the day of the cutoff - no future information.
    """
    all_students = activity[[COL_PID]].drop_duplicates()
    feature_columns = [
        "n_rows", "n_active_days", "n_lms", "n_video", "rows_per_active_day",
        "span_days", "weekly_mean", "weekly_std", "weekly_cv",
        "weekly_trend_slope", "active_weeks_first4", "night_ratio",
        "pct_gaps_under_15min", "max_continuous_run_hrs",
    ]
    df = activity[activity[COL_DATE] <= cutoff].copy()
    if df.empty:
        return all_students.assign(
            **{column: 0.0 for column in feature_columns}
        ).reset_index(drop=True)

    df["hour"] = df[COL_TS].dt.hour
    df["week"] = (df[COL_DATE] - AUTUMN_START).dt.days // 7

    rows = []

    g = df.groupby(COL_PID)
    for pid, grp in g:
        grp = grp.sort_values(COL_TS)
        n_rows = len(grp)
        n_active_days = grp[COL_DATE].nunique()
        n_lms = (grp["source"] == "LMS").sum()
        n_video = (grp["source"] == "VIDEO").sum()
        span_days = max((grp[COL_DATE].max() - grp[COL_DATE].min()).days, 1)

        weekly = grp.groupby("week").size()
        weekly_mean = weekly.mean() if len(weekly) else 0
        weekly_std = weekly.std(ddof=0) if len(weekly) > 1 else 0
        # early trend: slope of weekly counts (declining engagement = negative)
        if len(weekly) >= 2:
            x = weekly.index.values.astype(float)
            y = weekly.values.astype(float)
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        # night-activity ratio (early vampire signal) - logic B (prev-day boundary
        # irrelevant here, we just need a ratio, not a day-count threshold)
        night_rows = (grp["hour"] < 5).sum()
        night_ratio = night_rows / n_rows if n_rows else 0

        # gap-based "continuous activity" signal (early zombie signal)
        ts = grp[COL_TS].values
        gaps_min = np.diff(ts).astype("timedelta64[s]").astype(float) / 60 if len(ts) > 1 else np.array([])
        pct_gaps_under_15 = (gaps_min < 15).mean() if len(gaps_min) else 0
        max_continuous_run_hrs = 0
        if len(gaps_min):
            run = 0
            best = 0
            for gpd in gaps_min:
                if gpd <= 15:
                    run += gpd
                else:
                    best = max(best, run)
                    run = 0
            best = max(best, run)
            max_continuous_run_hrs = best / 60

        active_weeks_first4 = weekly[weekly.index.isin([0, 1, 2, 3])].count() if len(weekly) else 0

        rows.append({
            COL_PID: pid,
            "n_rows": n_rows,
            "n_active_days": n_active_days,
            "n_lms": n_lms,
            "n_video": n_video,
            "rows_per_active_day": n_rows / n_active_days if n_active_days else 0,
            "span_days": span_days,
            "weekly_mean": weekly_mean,
            "weekly_std": weekly_std,
            "weekly_cv": (weekly_std / weekly_mean) if weekly_mean else 0,  # spikiness proxy
            "weekly_trend_slope": slope,
            "active_weeks_first4": active_weeks_first4,
            "night_ratio": night_ratio,
            "pct_gaps_under_15min": pct_gaps_under_15,
            "max_continuous_run_hrs": max_continuous_run_hrs,
        })

    feat = pd.DataFrame(rows)
    # Keep every student that exists at all (even zero-activity-by-cutoff
    # students become all-zero rows, which is itself a meaningful signal).
    feat = all_students.merge(feat, on=COL_PID, how="left").fillna(0)
    return feat


# ==================================================================
# STEP 3 - LABELS (TRAINING DATA ONLY, uses the FULL academic year -
# this is the one place future information is allowed, because it's
# literally what we're trying to predict ahead of time)
# ==================================================================
def behavioural_monster_labels(activity: pd.DataFrame) -> pd.Series:
    """
    1 if the student is EVER flagged as Ghost, Zombie, Vampire, or Werewolf
    across the FULL training year. These are the only monsters for which
    "early warning sign" is a meaningful concept (data anomalies like
    Phantom/Doppelganger/Undead/Shape-Shifter are integrity issues, not
    behavioural trajectories you'd expect to see developing early).
    """
    df = activity.copy()

    # --- Ghost ---
    d = df.copy()
    year_end = d[COL_DATE].max()
    d = d[(d[COL_DATE] >= AUTUMN_START) & (d[COL_DATE] <= year_end)]
    d["week"] = (d[COL_DATE] - AUTUMN_START).dt.days // 7
    total_weeks = int((year_end - AUTUMN_START).days // 7) + 1
    n_later_weeks = max(total_weeks - 4, 1)
    weekly = d.groupby([COL_PID, "week"]).size().rename("n").reset_index()
    early = weekly[weekly["week"].between(0, 3)]
    active_early = early.groupby(COL_PID)["week"].nunique()
    early_avg = early.groupby(COL_PID)["n"].sum() / 4.0
    later_total = weekly[(weekly["week"] > 3) & (weekly["week"] < total_weeks)].groupby(COL_PID)["n"].sum()
    stats = pd.DataFrame({"active_early_weeks": active_early, "early_avg": early_avg}).fillna(0)
    stats["later_avg"] = later_total.reindex(stats.index).fillna(0) / n_later_weeks
    ghosts = set(stats[(stats["active_early_weeks"] >= 3) & (stats["early_avg"] > 0) &
                        (stats["later_avg"] < 0.10 * stats["early_avg"])].index)

    # --- Vampire (logic B: midnight-5am counts toward previous calendar date) ---
    d = df.copy()
    d["hour"] = d[COL_TS].dt.hour
    is_night = d["hour"] < 5
    night_date = d[COL_DATE] - pd.to_timedelta(is_night.astype(int), unit="D")
    d["night_bucket_date"] = np.where(is_night, night_date, d[COL_DATE])
    daily = d.groupby([COL_PID, "night_bucket_date"]).agg(
        any_night=("hour", lambda h: (h < 5).any()),
        any_day=("hour", lambda h: (h >= 5).any())).reset_index()
    agg = daily.groupby(COL_PID).agg(night_days=("any_night", "sum"), day_days=("any_day", "sum"))
    vampires = set(agg[(agg["night_days"] > agg["day_days"]) & (agg["night_days"] > 21)].index)

    # --- Zombie ---
    d = df.sort_values([COL_PID, COL_TS]).copy()
    d["gap_min"] = d.groupby(COL_PID)[COL_TS].diff().dt.total_seconds() / 60
    d["new_session"] = d["gap_min"].isna() | (d["gap_min"] > 15)
    d["session_id"] = d.groupby(COL_PID)["new_session"].cumsum()
    sess = d.groupby([COL_PID, "session_id"])[COL_TS].agg(start="min", end="max").reset_index()
    sess["hrs"] = (sess["end"] - sess["start"]).dt.total_seconds() / 3600
    zombies = set(sess[sess["hrs"] > 3][COL_PID].unique())

    # --- Werewolf (recurring large spikes in both terms) ---
    d = df[(df[COL_DATE] >= AUTUMN_START) & (df[COL_DATE] <= SPRING_END)].copy()
    d["week"] = d[COL_DATE].dt.to_period("W").apply(lambda p: p.start_time)
    weekly = d.groupby([COL_PID, "week"]).size().rename("n").reset_index()
    med = weekly.groupby(COL_PID)["n"].transform("median")
    mad = weekly.groupby(COL_PID)["n"].transform(lambda s: (s - s.median()).abs().median())
    mad_eff = mad.where(mad > 0, weekly.groupby(COL_PID)["n"].transform("std")).fillna(1).replace(0, 1)
    weekly["z"] = (weekly["n"] - med) / (1.4826 * mad_eff)
    weekly["med"] = med
    is_spike = (weekly["z"] > 4.0) & (weekly["n"] >= 10) & (weekly["n"] >= 3.0 * weekly["med"].clip(lower=1))
    weekly["term"] = np.where(weekly["week"] <= AUTUMN_END, "autumn",
                       np.where(weekly["week"] >= SPRING_START, "spring", "other"))
    spikes = weekly[is_spike & (weekly["term"] != "other")]
    per_term = spikes.groupby([COL_PID, "term"]).size().unstack(fill_value=0)
    werewolves = set(per_term[(per_term.get("autumn", 0) >= 2) & (per_term.get("spring", 0) >= 2)].index) \
        if len(per_term) else set()

    monster_pids = ghosts | vampires | zombies | werewolves
    print(f"[labels] Ghost={len(ghosts)} Vampire={len(vampires)} Zombie={len(zombies)} "
          f"Werewolf={len(werewolves)} -> union={len(monster_pids)} future-monster students")

    all_pids = df[COL_PID].unique()
    return pd.Series({pid: int(pid in monster_pids) for pid in all_pids}, name="is_future_monster")


# ==================================================================
# STEP 4 - MODEL TRAINING / SELECTION
# ==================================================================
def get_models():
    return {
        "LogisticRegression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=RANDOM_STATE)),
        ]),
        "GaussianNB": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", GaussianNB()),
        ]),
        "KNN": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ]),
        "SVM_RBF": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def _stratified_cv(y, max_splits=5):
    """Use as many stratified folds as the minority class can support."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError(
            "Stratified evaluation needs at least two examples of each class; "
            f"found class counts {counts.tolist()}."
        )
    return StratifiedKFold(
        n_splits=min(max_splits, int(counts.min())),
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def compare_models(X, y):
    """Compare models with ranking metrics and imbalance-aware threshold metrics."""
    cv = _stratified_cv(y)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "accuracy_at_0_5": "accuracy",
        "balanced_accuracy_at_0_5": "balanced_accuracy",
        "f1_at_0_5": "f1",
        "precision_at_0_5": "precision",
        "recall_at_0_5": "recall",
        "specificity_at_0_5": make_scorer(
            recall_score, pos_label=0, zero_division=0
        ),
        "mcc_at_0_5": "matthews_corrcoef",
        "brier_loss": "neg_brier_score",
    }
    results = []
    for name, pipe in get_models().items():
        scores = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
        )
        row = {"model": name, "cv_folds": cv.n_splits}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            if metric == "brier_loss":
                values = -values
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        results.append(row)
        print(
            f"[cv] {name:<20} PR-AUC={row['pr_auc_mean']:.3f}  "
            f"ROC-AUC={row['roc_auc_mean']:.3f}  "
            f"F1@0.5={row['f1_at_0_5_mean']:.3f}  "
            f"balanced-accuracy={row['balanced_accuracy_at_0_5_mean']:.3f}"
        )
    return pd.DataFrame(results).sort_values(
        "pr_auc_mean", ascending=False, ignore_index=True
    )


def select_f1_threshold(y_true, probabilities):
    """Choose a decision threshold from training-only out-of-fold probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        raise ValueError("Cannot tune a threshold without probability thresholds.")
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + np.finfo(float).eps
    )
    best_indices = np.flatnonzero(np.isclose(f1, f1.max()))
    return float(thresholds[best_indices[-1]])


def evaluate_holdout(y_true, probabilities, threshold, model_name):
    """Return holdout metrics at the training-selected threshold."""
    predictions = (probabilities >= threshold).astype(int)
    positives = int(np.sum(y_true))
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "threshold": threshold,
        "n_students": len(y_true),
        "positive_count": positives,
        "positive_rate": positives / len(y_true),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "pr_auc_baseline": positives / len(y_true),
        "brier_loss": brier_score_loss(y_true, probabilities),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def plot_model_comparison(comparison, output_path):
    """Save a compact comparison of ranking and thresholded CV metrics."""
    metrics = [
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("f1_at_0_5", "F1 @ 0.5"),
        ("balanced_accuracy_at_0_5", "Balanced accuracy @ 0.5"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4), sharey=True)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for ax, (metric, title), color in zip(axes, metrics, colors):
        means = comparison[f"{metric}_mean"]
        errors = comparison[f"{metric}_std"]
        ax.bar(comparison["model"], means, yerr=errors, color=color, capsize=3)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Cross-validation score (mean ± SD)")
    fig.suptitle("Training-fold model comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_holdout_diagnostics(y_true, probabilities, threshold, output_path):
    """Save precision-recall, ROC, and confusion-matrix views of the holdout."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    average_precision = average_precision_score(y_true, probabilities)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(recall, precision, color="#4c78a8", label=f"AP = {average_precision:.3f}")
    axes[0].axhline(np.mean(y_true), color="gray", linestyle="--", label="Positive rate")
    if len(thresholds):
        threshold_index = int(np.argmin(np.abs(thresholds - threshold)))
        axes[0].scatter(
            recall[threshold_index], precision[threshold_index],
            color="#e45756", label=f"Chosen threshold = {threshold:.3f}", zorder=3,
        )
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(fpr, tpr, color="#54a24b", label=f"ROC-AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.25)

    image = axes[2].imshow(matrix, cmap="Blues")
    axes[2].set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Not flagged", "Flagged"],
        yticklabels=["Not a monster", "Future monster"],
        xlabel="Predicted label", ylabel="Actual label", title="Holdout confusion matrix",
    )
    for row in range(2):
        for col in range(2):
            axes[2].text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Untouched holdout diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_dashboard(activity, test_features, y_true, probabilities,
                            threshold, output_path):
    """Create a four-panel early-warning and holdout-verification dashboard."""
    predictions = (probabilities >= threshold).astype(int)
    test_features = test_features.reset_index(drop=True).copy()
    test_features["actual"] = np.asarray(y_true)

    original_rc = plt.rcParams.copy()
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0b1220")
    for ax in axes.flat:
        ax.set_facecolor("#111827")
        ax.grid(alpha=0.16)

    # The model predicts the union of behavioral outcomes, not an individual
    # monster type, so the first panel reports the binary holdout counts.
    actual_positive = int(np.sum(y_true))
    flagged = int(predictions.sum())
    true_positive = int(np.sum((np.asarray(y_true) == 1) & (predictions == 1)))
    count_labels = ["Verified future\nmonsters", "Flagged by\npredictor", "Correctly\nflagged"]
    count_values = [actual_positive, flagged, true_positive]
    axes[0, 0].bar(
        count_labels, count_values,
        color=["#ec4899", "#f59e0b", "#06b6d4"],
    )
    axes[0, 0].set_title("1. Future Monsters: Prediction vs. Verified Holdout",
                         color="#38bdf8", fontweight="bold")
    axes[0, 0].set_ylabel("Students")
    for index, value in enumerate(count_values):
        axes[0, 0].text(index, value, str(value), ha="center", va="bottom")

    negatives = test_features["actual"] == 0
    positives = ~negatives
    axes[0, 1].scatter(
        test_features.loc[negatives, "n_active_days"],
        test_features.loc[negatives, "night_ratio"],
        s=14, alpha=0.35, color="#64748b", label="Other students",
    )
    axes[0, 1].scatter(
        test_features.loc[positives, "n_active_days"],
        test_features.loc[positives, "night_ratio"],
        s=45, alpha=0.9, color="#fb7185", edgecolors="white",
        linewidths=0.4, label="Verified future monsters",
    )
    axes[0, 1].set_title("2. Early Activity vs. Night-Activity Share",
                         color="#fb7185", fontweight="bold")
    axes[0, 1].set_xlabel("Active days by cutoff")
    axes[0, 1].set_ylabel("Night activity share")
    axes[0, 1].legend(loc="best", fontsize=8)

    test_pids = test_features[COL_PID]
    early = activity[
        activity[COL_PID].isin(test_pids) & (activity[COL_DATE] <= CUTOFF_DATE)
    ].copy()
    early["week"] = (early[COL_DATE] - AUTUMN_START).dt.days // 7
    last_week = int(early["week"].max()) if not early.empty else -1
    weeks = np.arange(0, max(last_week + 1, 1))
    weekly_counts = (
        early.groupby([COL_PID, "week"]).size().unstack(fill_value=0)
        .reindex(index=test_pids, columns=weeks, fill_value=0)
    )
    labels_by_pid = test_features.set_index(COL_PID)["actual"]
    for label, color, name in [
        (0, "#22c55e", "Other students"),
        (1, "#c084fc", "Verified future monsters"),
    ]:
        cohort_pids = labels_by_pid[labels_by_pid == label].index
        if len(cohort_pids):
            means = weekly_counts.reindex(cohort_pids).mean(axis=0)
            axes[1, 0].plot(
                weeks + 1, means, marker="o", linewidth=2,
                color=color, label=f"{name} (n={len(cohort_pids)})",
            )
    axes[1, 0].axvline(4.5, color="#facc15", linestyle="--",
                       label="End of first 4 weeks")
    axes[1, 0].set_title("3. Early-Term Activity Trajectory (Cutoff Only)",
                         color="#c084fc", fontweight="bold")
    axes[1, 0].set_xlabel("Autumn week")
    axes[1, 0].set_ylabel("Mean activity rows per student")
    axes[1, 0].legend(loc="best", fontsize=8)

    for label, color, name in [
        (0, "#94a3b8", "Other students"),
        (1, "#fb7185", "Verified future monsters"),
    ]:
        cohort = test_features["actual"] == label
        axes[1, 1].scatter(
            test_features.loc[cohort, "n_rows"],
            test_features.loc[cohort, "max_continuous_run_hrs"] * 60,
            color=color, s=18, alpha=0.5, label=name,
        )
    axes[1, 1].axhline(180, color="#f97316", linestyle="--",
                       label="3-hour session label criterion")
    axes[1, 1].set_title("4. Early Activity Volume vs. Continuous Session",
                         color="#fb923c", fontweight="bold")
    axes[1, 1].set_xlabel("Activity rows by cutoff")
    axes[1, 1].set_ylabel("Longest continuous session (minutes)")
    axes[1, 1].legend(loc="best", fontsize=8)

    fig.suptitle(
        "MINI MONSTER HACK — EARLY WARNING & HOLDOUT VERIFICATION",
        color="#f8fafc", fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(original_rc)


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    print(f"Cutoff date for ALL features: {CUTOFF_DATE.date()} (no data after this is ever used as a feature)\n")

    # ---- 1. Training data: full activity (for labels) + early-only features
    print("=== Loading TRAINING data ===")
    train_activity_full, train_person, train_pmi = build_activity(TRAIN_DIR)

    print("\n=== Engineering TRAINING features (cutoff-restricted) ===")
    train_features = engineer_features(train_activity_full, cutoff=CUTOFF_DATE)

    print("\n=== Computing TRAINING labels (full academic year) ===")
    labels = behavioural_monster_labels(train_activity_full)
    train_features = train_features.merge(
        labels.rename("is_future_monster"), left_on=COL_PID, right_index=True, how="left")
    train_features["is_future_monster"] = train_features["is_future_monster"].fillna(0).astype(int)
    train_features.to_csv(os.path.join(OUT_DIR, "training_labels.csv"), index=False)

    feature_cols = [c for c in train_features.columns if c not in (COL_PID, "is_future_monster")]
    X = train_features[feature_cols]
    y = train_features["is_future_monster"]
    print(f"\nTraining set: {len(X)} students, {y.sum()} positive ({y.mean()*100:.1f}%)")

    # ---- 2. Held-out evaluation split (students, stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    print("\n=== Cross-validating models on the TRAINING split ===")
    comparison = compare_models(X_tr, y_tr)
    comparison.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    plot_model_comparison(
        comparison, os.path.join(OUT_DIR, "model_comparison.png")
    )
    best_name = comparison.iloc[0]["model"]
    print(f"\n[selected] Best model by CV PR-AUC: {best_name}")

    best_pipe = get_models()[best_name]
    oof_probabilities = cross_val_predict(
        best_pipe,
        X_tr,
        y_tr,
        cv=_stratified_cv(y_tr),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    risk_threshold = select_f1_threshold(y_tr, oof_probabilities)
    best_pipe.fit(X_tr, y_tr)
    y_prob = best_pipe.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= risk_threshold).astype(int)

    print(
        f"\n[threshold] F1-optimal training OOF threshold: {risk_threshold:.4f}"
    )
    print("\n=== Held-out evaluation (threshold and model never fit on holdout) ===")
    print(classification_report(y_te, y_pred, zero_division=0))
    heldout_metrics = evaluate_holdout(y_te, y_prob, risk_threshold, best_name)
    pd.DataFrame([heldout_metrics]).to_csv(
        os.path.join(OUT_DIR, "heldout_metrics.csv"), index=False
    )
    plot_holdout_diagnostics(
        y_te, y_prob, risk_threshold,
        os.path.join(OUT_DIR, "heldout_diagnostics.png"),
    )
    plot_predictor_dashboard(
        train_activity_full,
        train_features.loc[X_te.index],
        y_te,
        y_prob,
        risk_threshold,
        os.path.join(OUT_DIR, "predictor_dashboard.png"),
    )
    print(
        f"ROC-AUC={heldout_metrics['roc_auc']:.3f}  "
        f"PR-AUC={heldout_metrics['pr_auc']:.3f} "
        f"(baseline={heldout_metrics['pr_auc_baseline']:.3f})  "
        f"F1={heldout_metrics['f1']:.3f}  "
        f"precision={heldout_metrics['precision']:.3f}  "
        f"recall={heldout_metrics['recall']:.3f}  "
        f"specificity={heldout_metrics['specificity']:.3f}  "
        f"MCC={heldout_metrics['mcc']:.3f}  "
        f"Brier={heldout_metrics['brier_loss']:.3f}"
    )

    # ---- 3. Refit best model on ALL training data before real prediction
    best_pipe.fit(X, y)

    # ---- 4. Predict set: SAME feature function, SAME cutoff
    print("\n=== Loading PREDICT data ===")
    predict_activity, predict_person, _ = build_activity(PREDICT_DIR)

    print("\n=== Engineering PREDICT features (cutoff-restricted) ===")
    predict_features = engineer_features(predict_activity, cutoff=CUTOFF_DATE)
    predict_features.to_csv(os.path.join(OUT_DIR, "predict_features.csv"), index=False)

    X_predict = predict_features[feature_cols]
    risk_score = best_pipe.predict_proba(X_predict)[:, 1]
    risk_flag = (risk_score >= risk_threshold).astype(int)

    results = pd.DataFrame({
        COL_PID: predict_features[COL_PID],
        "risk_score": risk_score,
        "risk_threshold": risk_threshold,
        "predicted_monster_risk": risk_flag,
    }).sort_values("risk_score", ascending=False)

    out_path = os.path.join(OUT_DIR, "predict_results.csv")
    results.to_csv(out_path, index=False)

    print(f"\n[saved] {out_path}")
    print(f"[summary] {results['predicted_monster_risk'].sum()} / {len(results)} "
          f"students flagged as at-risk (model={best_name})")
