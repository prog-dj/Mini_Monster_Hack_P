"""
MINI MONSTER HACK - ALL IN ONE
==============================
Put this script in the SAME FOLDER as your raw CSVs. Required:

  la_mmh_person_dim.csv
  la_mmh_person_module_instance_fact.csv
  la_mmh_video_dim.csv
  la_mmh_video_activity_by_view_fact.csv
  la_mmh_lms_item_activity_by_session_fact.csv

Optional (used if present, skipped gracefully if not):
  la_mmh_calendar_dim.csv
  la_mmh_module_instance_dim.csv
  la_mmh_lms_item_dim.csv

Then run:
    python monster_hack_all_in_one.py

It will:
  1. Load + merge everything into one activity table + one programme table
  2. Run all 9 monster detectors from the Monster Guide
  3. Save combined_activity.csv, combined_programmes.csv, combined_students.csv
  4. Save result_<monster>.csv for each monster (full evidence/detail)
  5. Save submission.csv  ->  P_ID, Monster_Type  (official submission format)

ADJUST BEFORE TRUSTING RESULTS:
  - AUTUMN_START / AUTUMN_END / SPRING_START / SPRING_END below (Ghost, Werewolf)
  - MMH_DATA_DIR env var if your CSVs live in a different folder

NOTE ON BIG FILES: the LMS/video activity fact files can be hundreds of MB.
This script only reads the columns it actually needs and uses compact dtypes
to keep memory usage reasonable.
"""

import os
import pandas as pd
import numpy as np

# ==================================================================
# CONFIG
# ==================================================================
DATA_DIR = os.environ.get("MMH_DATA_DIR", ".")

# Keywords used to FIND each file by fuzzy match, so it doesn't matter if your
# files are named la_mmh_person_dim.csv, la_mmh_training_person_dim.csv,
# person_dim_v2.csv, etc. All keywords in a tuple must appear (case-insensitive).
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

# Files that are nice-to-have but not required to run the detectors.
OPTIONAL_FILES = {"calendar", "module_inst", "lms_dim"}

# Only pull the columns each big fact file actually needs (saves a lot of
# memory/time on multi-hundred-MB files). None = read all columns.
USECOLS = {
    "lms_fact":   ["P_ID", "DATE", "TIMESTAMP", "LMS_ITEM_ID", "INTERACTION_COUNT"],
    "video_fact": ["P_ID", "DATE", "TIMESTAMP", "VIDEO_ID", "TOTAL_MINUTES_DELIVERED"],
}

# Compact dtypes for the big files (cuts memory roughly in half vs defaults).
DTYPES = {
    "lms_fact":   {"P_ID": "category", "LMS_ITEM_ID": "category",
                   "INTERACTION_COUNT": "float32"},
    "video_fact": {"P_ID": "category", "VIDEO_ID": "category",
                   "TOTAL_MINUTES_DELIVERED": "float32"},
}
PARSE_DATES = {
    "lms_fact":   ["DATE", "TIMESTAMP"],
    "video_fact": ["DATE", "TIMESTAMP"],
}

NA_VALUES = ["null", "NULL", "", "NaN", "nan"]

COL_PID    = "P_ID"
COL_DATE   = "DATE"
COL_TS     = "TIMESTAMP"
COL_SOURCE = "source"       # "LMS" or "VIDEO"
COL_ITEM   = "item_id"
COL_DEREG  = "deregistration_date"

# Academic calendar - official term boundaries
AUTUMN_START = pd.Timestamp("2023-10-02")
AUTUMN_END   = pd.Timestamp("2024-01-06")
SPRING_START = pd.Timestamp("2024-01-07")
SPRING_END   = pd.Timestamp("2024-04-27")

MONSTER_LABELS = {
    "phantom": "Phantom",
    "ghost": "Ghost",
    "zombie": "Zombie",
    "vampire": "Vampire",
    "undead": "Undead Student",
    "doppelganger": "Doppelganger",
    "werewolf": "Werewolf",
    "web_crawler": "Web Crawler",
    "shape_shifter": "Shape-Shifter",
}


# ==================================================================
# STEP 1 - LOAD + COMBINE RAW CSVS
# ==================================================================
def _find_csv(keywords):
    """Search DATA_DIR for a .csv whose lowercase filename contains ALL keywords."""
    all_csvs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    matches = [
        f for f in all_csvs
        if all(kw.lower() in f.lower() for kw in keywords)
    ]
    return matches, all_csvs


def load(name, required=True):
    keywords = FILE_KEYWORDS[name]
    matches, all_csvs = _find_csv(keywords)

    if len(matches) == 0:
        if not required:
            print(f"[skip] no file found for optional '{name}' (keywords={keywords})")
            return None
        raise FileNotFoundError(
            f"\nCould not find a CSV matching keywords {keywords} for '{name}' "
            f"in folder '{os.path.abspath(DATA_DIR)}'.\n"
            f"CSV files found in that folder:\n  "
            + ("\n  ".join(all_csvs) if all_csvs else "(none)")
            + f"\n\nFix: rename/move the right file so its name contains {keywords}, "
              f"or set MMH_DATA_DIR to the correct folder."
        )
    if len(matches) > 1:
        print(f"[warn] multiple files match {keywords} for '{name}': {matches} "
              f"-> using '{matches[0]}'")

    path = os.path.join(DATA_DIR, matches[0])
    size_mb = os.path.getsize(path) / 1e6

    read_kwargs = dict(na_values=NA_VALUES)
    if name in USECOLS:
        read_kwargs["usecols"] = lambda c: c in USECOLS[name]
    if name in DTYPES:
        read_kwargs["dtype"] = DTYPES[name]
    if name in PARSE_DATES:
        read_kwargs["parse_dates"] = PARSE_DATES[name]

    print(f"[loading] {matches[0]} ({size_mb:,.0f} MB) ...")
    df = pd.read_csv(path, **read_kwargs)
    print(f"[loaded]  {matches[0]:55s} shape={df.shape}")
    return df


def combine_data():
    person      = load("person")
    pmi_fact    = load("pmi_fact")
    video_dim   = load("video_dim")
    video_fact  = load("video_fact")
    lms_fact    = load("lms_fact")

    # Optional lookup tables - script works fine without these.
    calendar    = load("calendar", required=False)
    module_inst = load("module_inst", required=False)
    lms_dim     = load("lms_dim", required=False)

    def _to_naive_datetime(s):
        s = pd.to_datetime(s, errors="coerce", utc=True)
        try:
            if s.dt.tz is not None:
                s = s.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        return s

    for c in ["registration_date", "deregistration_date"]:
        if c in person.columns:
            person[c] = _to_naive_datetime(person[c])
    if calendar is not None and "DATE" in calendar.columns:
        calendar["DATE"] = _to_naive_datetime(calendar["DATE"])
    for df_ in (video_fact, lms_fact):
        for c in ["DATE", "TIMESTAMP"]:
            if c in df_.columns:
                df_[c] = _to_naive_datetime(df_[c])

    # LMS activity, optionally enriched with item + module info if available
    lms = lms_fact.copy()
    if lms_dim is not None:
        lms = lms.merge(lms_dim, on="LMS_ITEM_ID", how="left")
    if module_inst is not None and "MODULE_INSTANCE_ID" in lms.columns:
        lms = lms.merge(module_inst, on="MODULE_INSTANCE_ID", how="left")
    lms["source"] = "LMS"
    lms["item_id"] = lms["LMS_ITEM_ID"]
    lms["activity_value"] = lms["INTERACTION_COUNT"]

    # Video activity, optionally enriched with video + module info if available
    vid = video_fact.merge(video_dim, on="VIDEO_ID", how="left")
    if module_inst is not None and "MODULE_INSTANCE_ID" in vid.columns:
        vid = vid.merge(module_inst, on="MODULE_INSTANCE_ID", how="left")
    vid["source"] = "VIDEO"
    vid["item_id"] = vid["VIDEO_ID"]
    vid["activity_value"] = vid["TOTAL_MINUTES_DELIVERED"]

    # Stack into one long activity table - only using columns that exist
    common_cols = [
        "P_ID", "DATE", "TIMESTAMP", "source", "item_id", "activity_value",
        "MODULE_INSTANCE_ID", "MODULE_LEVEL", "ACADEMIC_YEAR_START", "TERM",
        "DEPARTMENT_ID",
    ]
    lms_slim = lms.reindex(columns=common_cols)
    vid_slim = vid.reindex(columns=common_cols)
    del lms, vid  # free memory before concat of large frames

    activity = pd.concat([lms_slim, vid_slim], ignore_index=True)
    del lms_slim, vid_slim
    activity = activity.sort_values(["P_ID", "TIMESTAMP"]).reset_index(drop=True)

    if calendar is not None:
        cal_cols = [c for c in ["DATE", "WEEK_NUMBER", "ACADEMIC_YEAR"] if c in calendar.columns]
        activity = activity.merge(calendar[cal_cols], on="DATE", how="left")
    else:
        # Derive a week number ourselves so Ghost/Werewolf still work.
        activity["WEEK_NUMBER"] = activity["DATE"].dt.isocalendar().week

    activity = activity.merge(person, on="P_ID", how="left")

    print(f"[combined] activity table shape={activity.shape}")

    # Programme table for Shape-Shifter (kept separate -- merging directly
    # into activity would duplicate every row per concurrent module).
    # Falls back to MODULE_INSTANCE_ID as the "programme" proxy if no
    # module_instance_dim / DEPARTMENT_ID is available.
    programmes = pmi_fact.copy()
    if module_inst is not None:
        programmes = programmes.merge(module_inst, on="MODULE_INSTANCE_ID", how="left")
    if "DEPARTMENT_ID" not in programmes.columns:
        programmes["DEPARTMENT_ID"] = programmes["MODULE_INSTANCE_ID"]
        print("[note] no module_instance_dim/DEPARTMENT_ID found - using "
              "MODULE_INSTANCE_ID as the Shape-Shifter programme proxy.")
    print(f"[combined] programmes table shape={programmes.shape}")

    activity.to_csv("combined_activity.csv", index=False)
    programmes.to_csv("combined_programmes.csv", index=False)
    person.to_csv("combined_students.csv", index=False)
    print("[saved] combined_activity.csv, combined_programmes.csv, combined_students.csv\n")

    return activity, programmes, person


# ==================================================================
# STEP 2 - DETECTORS
# ==================================================================
def _prep(activity: pd.DataFrame) -> pd.DataFrame:
    df = activity.copy()
    df[COL_TS] = pd.to_datetime(df[COL_TS], errors="coerce")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    return df.dropna(subset=[COL_TS, COL_PID]).sort_values([COL_PID, COL_TS])


def find_phantom(activity):
    df = _prep(activity)
    key_cols = [COL_PID, COL_SOURCE, COL_ITEM, COL_TS]
    dupes = df[df.duplicated(subset=key_cols, keep=False)]
    return (dupes.groupby(COL_PID).size()
                 .reset_index(name="duplicate_rows")
                 .sort_values("duplicate_rows", ascending=False))


def find_ghost(activity, autumn_start=AUTUMN_START, year_end=None):
    """
    Active >=3 of the first 4 Autumn weeks, then goes quiet for the rest of
    the academic year: later weekly average activity-row count < 10% of the
    early weekly average.

    IMPORTANT: "weekly average" means total rows / number of weeks in that
    period - INCLUDING weeks with zero activity. Averaging only over weeks
    that happen to have activity (the old approach) ignores silence and
    understates how quiet the later period really was, under-counting Ghosts.
    """
    df = _prep(activity)

    if year_end is None:
        year_end = df[COL_DATE].max()  # falls back to latest date in the data
    df = df[(df[COL_DATE] >= autumn_start) & (df[COL_DATE] <= year_end)]

    df["week"] = (df[COL_DATE] - autumn_start).dt.days // 7
    total_weeks = int((year_end - autumn_start).days // 7) + 1
    n_later_weeks = max(total_weeks - 4, 1)  # weeks 4 .. end, including zero-activity weeks

    weekly = df.groupby([COL_PID, "week"]).size().rename("n").reset_index()

    early = weekly[weekly["week"].between(0, 3)]
    active_early_weeks = early.groupby(COL_PID)["week"].nunique()
    early_total = early.groupby(COL_PID)["n"].sum()
    early_avg = early_total / 4.0  # fixed 4-week window

    later = weekly[(weekly["week"] > 3) & (weekly["week"] < total_weeks)]
    later_total = later.groupby(COL_PID)["n"].sum()

    stats = pd.DataFrame({
        "active_early_weeks": active_early_weeks,
        "early_avg": early_avg,
    })
    stats["later_total"] = later_total
    stats = stats.fillna(0)
    stats["later_avg"] = stats["later_total"] / n_later_weeks

    flagged = stats[
        (stats["active_early_weeks"] >= 3) &
        (stats["early_avg"] > 0) &
        (stats["later_avg"] < 0.10 * stats["early_avg"])
    ].drop(columns="later_total").reset_index()
    return flagged


def find_zombie(activity, max_gap_min=15, min_hours=3):
    df = _prep(activity)
    df["gap_min"] = df.groupby(COL_PID)[COL_TS].diff().dt.total_seconds() / 60
    df["new_session"] = df["gap_min"].isna() | (df["gap_min"] > max_gap_min)
    df["session_id"] = df.groupby(COL_PID)["new_session"].cumsum()

    sessions = (df.groupby([COL_PID, "session_id"])[COL_TS]
                  .agg(start="min", end="max")
                  .reset_index())
    sessions["duration_hrs"] = (sessions["end"] - sessions["start"]).dt.total_seconds() / 3600

    long_sessions = sessions[sessions["duration_hrs"] > min_hours]
    return (long_sessions.groupby(COL_PID)["duration_hrs"]
                         .max()
                         .reset_index(name="longest_session_hrs")
                         .sort_values("longest_session_hrs", ascending=False))


def find_vampire(activity, min_night_days=21):
    df = _prep(activity)
    df["hour"] = df[COL_TS].dt.hour
    df["is_night"] = df["hour"] < 5  # > midnight to 5am

    daily = df.groupby([COL_PID, COL_DATE])["is_night"].any().reset_index()
    daily["is_day_activity"] = ~daily["is_night"]

    agg = daily.groupby(COL_PID).agg(
        night_days=("is_night", "sum"),
        day_days=("is_day_activity", "sum"),
    ).reset_index()

    return agg[(agg["night_days"] > agg["day_days"]) &
               (agg["night_days"] > min_night_days)].copy()


def find_undead(activity):
    df = _prep(activity)
    df[COL_DEREG] = pd.to_datetime(df[COL_DEREG], errors="coerce")
    valid = df.dropna(subset=[COL_DEREG])
    violations = valid[valid[COL_TS] > valid[COL_DEREG]]

    return (violations.groupby(COL_PID)
                      .agg(deregistration_date=(COL_DEREG, "first"),
                           violations=(COL_TS, "count"),
                           last_activity=(COL_TS, "max"))
                      .reset_index()
                      .sort_values("violations", ascending=False))


def find_doppelganger(activity):
    df = _prep(activity)
    df["event"] = (df[COL_SOURCE].astype(str) + "|" +
                    df[COL_ITEM].astype(str) + "|" +
                    df[COL_TS].astype(str))

    sig = (df.groupby(COL_PID)["event"]
             .apply(lambda s: hash(tuple(sorted(s))))
             .reset_index(name="signature"))

    dup_sigs = sig[sig.duplicated("signature", keep=False)]
    return dup_sigs.sort_values("signature").reset_index(drop=True)


def find_werewolf(activity, autumn=(AUTUMN_START, AUTUMN_END),
                   spring=(SPRING_START, SPRING_END), z_thresh=4.0,
                   min_abs_count=10, min_ratio=3.0, min_spike_weeks_per_term=2):
    """
    A real Werewolf has RECURRING, LARGE spikes in BOTH Autumn and Spring -
    not just one statistically-unusual week.

    IMPORTANT FIX: a pure z-score (even robust median/MAD) blows up for
    students whose weekly counts are naturally small and tight (e.g. median
    2-3 events/week, MAD ~0.5) - a week with just 2-3 MORE events than usual
    can produce z > 3 even though the raw jump is trivial. That was flagging
    ~6% of ordinary low-activity students as "spiky" on real-world data.

    Fix: a week only counts as a genuine spike if it clears ALL three bars:
      1. z-score > z_thresh          (statistically unusual for this student)
      2. raw count >= min_abs_count  (a trivial absolute floor - no spike
                                       should be built from single-digit counts)
      3. raw count >= min_ratio x their own median week (a real multiple of
                                       their normal pace, not a +2 nudge)
    And it must happen on >=2 separate weeks in EACH term (recurring
    "throughout the term", not a single one-off blip in one term).
    """
    df = _prep(activity)
    df = df[(df[COL_DATE] >= autumn[0]) & (df[COL_DATE] <= spring[1])]
    df["week"] = df[COL_DATE].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = df.groupby([COL_PID, "week"]).size().rename("n").reset_index()

    g = weekly.groupby(COL_PID)["n"]
    med = g.transform("median")
    mad = weekly.groupby(COL_PID)["n"].transform(lambda s: (s - s.median()).abs().median())
    mad_eff = mad.where(mad > 0, g.transform("std")).fillna(1).replace(0, 1)
    weekly["z"] = (weekly["n"] - med) / (1.4826 * mad_eff)
    weekly["med"] = med

    is_spike = (
        (weekly["z"] > z_thresh) &
        (weekly["n"] >= min_abs_count) &
        (weekly["n"] >= min_ratio * weekly["med"].clip(lower=1))
    )
    weekly["term"] = np.where(weekly["week"] <= autumn[1], "autumn",
                       np.where(weekly["week"] >= spring[0], "spring", "other"))

    spikes = weekly[is_spike & (weekly["term"] != "other")]

    per_term = spikes.groupby([COL_PID, "term"]).size().unstack(fill_value=0)
    qualifying = per_term[
        (per_term.get("autumn", 0) >= min_spike_weeks_per_term) &
        (per_term.get("spring", 0) >= min_spike_weeks_per_term)
    ].index

    return (spikes[spikes[COL_PID].isin(qualifying)]
             .groupby(COL_PID)
             .agg(spike_weeks=("week", "nunique"), max_z=("z", "max"),
                  max_weekly_count=("n", "max"))
             .reset_index()
             .sort_values("max_z", ascending=False))


def find_web_crawler(activity, window_hours=12, threshold=200):
    df = _prep(activity)
    results = []

    for pid, grp in df.groupby(COL_PID):
        for source in ("LMS", "VIDEO"):
            sub = grp[grp[COL_SOURCE] == source].set_index(COL_TS).sort_index()
            if len(sub) <= threshold:
                continue
            peak = 0
            ts = sub.index.to_numpy()
            items = sub[COL_ITEM].to_numpy()
            left = 0
            counts = {}
            for right in range(len(ts)):
                counts[items[right]] = counts.get(items[right], 0) + 1
                while ts[right] - ts[left] > np.timedelta64(window_hours, "h"):
                    counts[items[left]] -= 1
                    if counts[items[left]] == 0:
                        del counts[items[left]]
                    left += 1
                peak = max(peak, len(counts))
            if peak > threshold:
                results.append({COL_PID: pid, "source": source, "peak_distinct_items_12h": peak})

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values("peak_distinct_items_12h", ascending=False)
    return out


def find_shape_shifter(programmes):
    """
    Flags students who appear on >1 DISTINCT PROGRAMME at DIFFERENT TIMES
    (a genuine switch), NOT students simply taking several modules at once
    in the same term (completely normal - most students do this).

    Requires a real programme/department field plus a time axis
    (ACADEMIC_YEAR_START, TERM), which only exist if module_instance_dim.csv
    was supplied. Without it, MODULE_INSTANCE_ID was previously used as a
    fake "programme" proxy - but since students are concurrently enrolled
    on many module instances every term, that flagged almost everyone
    (this is the bug that produced ~16,000 false "monsters"). We now
    refuse to guess and return an empty result with a loud warning instead.
    """
    has_dept = "DEPARTMENT_ID" in programmes.columns and programmes["DEPARTMENT_ID"].notna().any()
    has_time = "ACADEMIC_YEAR_START" in programmes.columns and "TERM" in programmes.columns

    if not (has_dept and has_time):
        print("[WARNING] Shape-Shifter needs la_mmh_module_instance_dim.csv "
              "(for a real DEPARTMENT_ID + ACADEMIC_YEAR_START/TERM). "
              "Without it there is no reliable way to tell 'several modules "
              "in one term' (normal) apart from 'switched programme over "
              "time' (the actual monster). Returning 0 results instead of "
              "guessing - get that file before submitting Shape-Shifter.")
        return pd.DataFrame(columns=[COL_PID, "n_departments", "departments"])

    prog = programmes.dropna(subset=["DEPARTMENT_ID", "ACADEMIC_YEAR_START", "TERM"]).copy()
    prog = prog.sort_values([COL_PID, "ACADEMIC_YEAR_START", "TERM"])

    # A real switch = the department seen in a LATER term differs from the
    # department(s) seen in an EARLIER term for the same student - not just
    # "more than one department ever" (that conflates concurrent + sequential).
    def switched(group):
        ordered_terms = group[["ACADEMIC_YEAR_START", "TERM"]].drop_duplicates().sort_values(
            ["ACADEMIC_YEAR_START", "TERM"])
        depts_by_term = [
            set(group[(group["ACADEMIC_YEAR_START"] == t.ACADEMIC_YEAR_START) &
                      (group["TERM"] == t.TERM)]["DEPARTMENT_ID"])
            for t in ordered_terms.itertuples()
        ]
        # flag only if some later term's dept set has NO overlap with an earlier one
        for i in range(1, len(depts_by_term)):
            if depts_by_term[i].isdisjoint(set.union(*depts_by_term[:i])):
                return True
        return False

    flags = prog.groupby(COL_PID).filter(switched)[COL_PID].unique()

    detail = (prog[prog[COL_PID].isin(flags)]
                .groupby(COL_PID)["DEPARTMENT_ID"]
                .apply(lambda s: list(dict.fromkeys(s)))
                .reset_index(name="departments"))
    detail["n_departments"] = detail["departments"].apply(len)
    return detail[[COL_PID, "n_departments", "departments"]]


def run_all(activity, programmes):
    results = {
        "phantom":       find_phantom(activity),
        "ghost":         find_ghost(activity),
        "zombie":        find_zombie(activity),
        "vampire":       find_vampire(activity),
        "undead":        find_undead(activity),
        "doppelganger":  find_doppelganger(activity),
        "werewolf":      find_werewolf(activity),
        "web_crawler":   find_web_crawler(activity),
        "shape_shifter": find_shape_shifter(programmes),
    }

    print("=== MONSTER SCAN SUMMARY ===")
    submission_rows = []
    for key, df in results.items():
        label = MONSTER_LABELS[key]
        n = df[COL_PID].nunique() if COL_PID in df.columns else 0
        print(f"{label:<16} -> {n} students flagged")
        if COL_PID in df.columns:
            for pid in df[COL_PID].drop_duplicates():
                submission_rows.append({"P_ID": pid, "Monster_Type": label})

    submission = pd.DataFrame(submission_rows).drop_duplicates()
    return results, submission


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    activity, programmes, students = combine_data()

    results, submission = run_all(activity, programmes)

    for key, df in results.items():
        fname = f"result_{key}.csv"
        df.to_csv(fname, index=False)
        print(f"[saved] {fname}")

    submission.to_csv("submission.csv", index=False)
    print("\n[saved] submission.csv  (P_ID, Monster_Type) - ready to paste into your Excel")

