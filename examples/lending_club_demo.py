"""Lending Club — Evidence Sufficiency Calculator evaluation (credit scoring).

Computes S(t) trajectories on Lending Club data across yearly windows
with three drift injection scenarios. Companion to Governance Drift Toolkit Lending Club demo.

Prerequisites
-------------
1. Install demo dependencies::

       pip install -e ".[demo]"

2. Clone the separate Governance Drift Toolkit repository next to this repo.

3. Prepare the Lending Club demo data there. The raw dataset comes from
   Kaggle and the download steps are documented in that repository.

4. This example expects the CSV at::

       ../governance-drift-toolkit/data/lending_club/accepted_2007_to_2018Q4.csv

5. Typical local setup::

    cd ../governance-drift-toolkit
    python examples/lending_club_demo.py

6. Run::

       python examples/lending_club_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("../governance-drift-toolkit/data/lending_club/accepted_2007_to_2018Q4.csv")
SEED = 42

FEATURE_COLS = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
    "revol_bal",
    "total_acc",
    "open_acc",
    "pub_rec",
    "installment",
]

SHIFT_FEATURES = ["annual_inc", "dti", "revol_util"]
COVARIATE_SIGMAS = [0.3, 0.6, 1.0, 1.5, 2.0]
CONCEPT_FLIP_RATES = [0.03, 0.06, 0.10, 0.15, 0.25]
MIXED_SIGMAS = [0.2, 0.4, 0.7, 1.0, 1.5]
MIXED_FLIPS = [0.02, 0.04, 0.08, 0.12, 0.20]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_and_split() -> tuple[list[pd.DataFrame], list[str]]:
    """Load Lending Club, create binary label, split by year."""
    import pandas as pd

    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        print("This example expects a sibling checkout of governance-drift-toolkit.")
        print("Prepare the Lending Club dataset there first, then rerun this demo.")
        print("Typical setup:")
        print("  cd ../governance-drift-toolkit && python examples/lending_club_demo.py")
        sys.exit(1)

    print("Loading Lending Club accepted loans ...")
    df = pd.read_csv(DATA_PATH, usecols=["issue_d", "loan_status", *FEATURE_COLS], low_memory=False)

    default_statuses = {"Charged Off", "Default", "Late (31-120 days)"}
    df = df[df["loan_status"].isin({"Fully Paid", *default_statuses})].copy()
    df["is_default"] = df["loan_status"].isin(default_statuses).astype(int)
    df = df.drop(columns=["loan_status"])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df = df.dropna(subset=["issue_d"]).sort_values("issue_d")

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["year"] = df["issue_d"].dt.year
    windows, labels = [], []
    for y in sorted(df["year"].unique()):
        wdf = df[df["year"] == y].copy()
        if len(wdf) > 1000:
            windows.append(wdf)
            labels.append(str(y))

    total = sum(len(w) for w in windows)
    print(f"  {total:,} loans, {len(windows)} yearly windows ({labels[0]}-{labels[-1]})")
    return windows, labels


# ---------------------------------------------------------------------------
# Drift injection (same as Governance Drift Toolkit for consistency)
# ---------------------------------------------------------------------------


def _inject_covariate(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    sigma = COVARIATE_SIGMAS[min(idx, len(COVARIATE_SIGMAS) - 1)]
    for col in SHIFT_FEATURES:
        out[col] = out[col] + rng.normal(0, sigma * out[col].std(), len(out))
    return out


def _inject_concept(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    flip_rate = CONCEPT_FLIP_RATES[min(idx, len(CONCEPT_FLIP_RATES) - 1)]
    n_flip = int(len(out) * flip_rate)
    flip_idx = rng.choice(out.index, size=n_flip, replace=False)
    out.loc[flip_idx, "is_default"] = 1 - out.loc[flip_idx, "is_default"]
    return out


def _inject_mixed(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> pd.DataFrame:
    out = _inject_covariate(df, idx, rng)
    flip_rate = MIXED_FLIPS[min(idx, len(MIXED_FLIPS) - 1)]
    n_flip = int(len(out) * flip_rate)
    flip_idx = rng.choice(out.index, size=n_flip, replace=False)
    out.loc[flip_idx, "is_default"] = 1 - out.loc[flip_idx, "is_default"]
    return out


# ---------------------------------------------------------------------------
# Sufficiency computation
# ---------------------------------------------------------------------------

_WIDTH = 115


def _compute_and_print(
    title: str,
    windows: list[pd.DataFrame],
    labels: list[str],
    model: object,
    scaler: object,
    ref_df: pd.DataFrame,
    inject_fn: object | None = None,
) -> list[dict]:
    """Compute S(t) per window using sufficiency dimension scorers."""
    from sklearn.metrics import f1_score as sk_f1

    from sufficiency import compute_sufficiency, credit_scoring_config
    from sufficiency.dimensions.completeness import compute_completeness
    from sufficiency.dimensions.freshness import compute_freshness
    from sufficiency.dimensions.reliability import compute_reliability
    from sufficiency.dimensions.representativeness import compute_representativeness

    config = credit_scoring_config()
    rng = np.random.default_rng(SEED)

    ref_scores = model.predict_proba(scaler.transform(ref_df[FEATURE_COLS].values))[:, 1]  # type: ignore[union-attr]

    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")
    print(
        f"{'Year':>6} | {'Loans':>8} | {'Def%':>5} | "
        f"{'C':>5} | {'F':>5} | {'R':>5} | {'P':>5} | "
        f"{'A(t)':>5} | {'S(t)':>5} | {'Status':>12} | {'F1':>5}"
    )
    print(f"{'-' * _WIDTH}")

    rows = []
    for i, raw_df in enumerate(windows[1:]):
        current_df = inject_fn(raw_df, i, rng) if inject_fn is not None else raw_df

        # Years since reference as proxy for label delay
        years_elapsed = i + 1
        delta_t_days = years_elapsed * 365.0

        # Completeness: simulate decreasing label availability over time
        # Credit scoring: loans mature over 36-60 months
        label_avail = max(0.3, 1.0 - years_elapsed * 0.06)
        n_total = len(current_df)
        n_labeled = int(n_total * label_avail)

        # Reliability: F1 on labeled subset
        y_true = current_df["is_default"].values[:n_labeled]
        x_cur = scaler.transform(current_df[FEATURE_COLS].values[:n_labeled])  # type: ignore[union-attr]
        y_pred = (model.predict_proba(x_cur)[:, 1] > 0.5).astype(int)  # type: ignore[union-attr]
        actual_f1 = float(sk_f1(y_true, y_pred)) if len(y_true) > 10 else 0.5

        # Representativeness: KS between ref and current scores
        cur_scores = model.predict_proba(  # type: ignore[union-attr]
            scaler.transform(current_df[FEATURE_COLS].values)  # type: ignore[union-attr]
        )[:, 1]

        dims = {
            "completeness": compute_completeness(n_labeled, n_total),
            "freshness": compute_freshness(delta_t_days, config.lambda_freshness),
            "reliability": compute_reliability(y_true, y_pred, rng_seed=SEED),
            "representativeness": compute_representativeness(ref_scores, cur_scores),
        }
        result = compute_sufficiency(dims, config)

        row = {
            "year": labels[i + 1],
            "n_loans": n_total,
            "default_rate": current_df["is_default"].mean(),
            "C": dims["completeness"].value,
            "F": dims["freshness"].value,
            "R": dims["reliability"].value,
            "P": dims["representativeness"].value,
            "gate": result.gate,
            "S_t": result.composite,
            "status": result.status.value,
            "actual_f1": actual_f1,
            "scenario": title,
        }
        rows.append(row)

        print(
            f"{labels[i + 1]:>6} | {n_total:>8,} | {current_df['is_default'].mean():>4.2f} | "
            f"{row['C']:>5.3f} | {row['F']:>5.3f} | {row['R']:>5.3f} | {row['P']:>5.3f} | "
            f"{row['gate']:>5.3f} | {row['S_t']:>5.3f} | {row['status']:>12} | {row['actual_f1']:>5.3f}"
        )

    return rows


# ---------------------------------------------------------------------------
# Blind period simulation
# ---------------------------------------------------------------------------


def _run_blind_period_sim() -> None:
    """Run BlindPeriodSimulator with credit scoring config."""
    from sufficiency import BlindPeriodSimulator, DriftSpec, DriftType, credit_scoring_config

    config = credit_scoring_config()

    print(f"\n{'=' * _WIDTH}")
    print("  Blind Period Simulation (credit_scoring_config, lambda=0.005)")
    print(f"{'=' * _WIDTH}")

    for name, specs in [
        ("No drift", []),
        ("Covariate P(X)", [DriftSpec(DriftType.COVARIATE, magnitude=0.6, onset_day=15)]),
        ("Concept P(Y|X)", [DriftSpec(DriftType.REAL_CONCEPT, magnitude=0.8, onset_day=10)]),
        (
            "Mixed",
            [
                DriftSpec(DriftType.COVARIATE, magnitude=0.4, onset_day=15),
                DriftSpec(DriftType.REAL_CONCEPT, magnitude=0.5, onset_day=20),
            ],
        ),
    ]:
        sim = BlindPeriodSimulator(
            initial_completeness=0.90,
            initial_reliability=0.75,
            initial_representativeness=0.92,
            config=config,
            drift_specs=specs,
        )
        print(f"\n  {name}:")
        for r in sim.simulate([30, 60, 90, 180, 365]):
            days = (r.timestamp - sim.start_time).days
            print(f"    Day {days:3d}: S={r.composite:.3f}  A={r.gate:.3f}  {r.status.value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run sufficiency evaluation on Lending Club."""
    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("Missing dependencies. Install with: pip install -e '.[demo]'")
        sys.exit(1)

    from sklearn.preprocessing import StandardScaler

    windows, labels = _load_and_split()

    ref_df = windows[0]
    scaler = StandardScaler()
    x_ref = scaler.fit_transform(ref_df[FEATURE_COLS].values)
    y_ref = ref_df["is_default"].values
    model = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")
    model.fit(x_ref, y_ref)
    print(f"  Reference: {labels[0]}, {len(ref_df):,} loans, default_rate={y_ref.mean():.4f}")

    all_rows: list[dict] = []
    for title, fn in [
        ("Baseline (natural drift)", None),
        ("Covariate Drift P(X)", _inject_covariate),
        ("Mixed Drift P(X)+P(Y|X)", _inject_mixed),
        ("Pure Concept Drift P(Y|X)", _inject_concept),
    ]:
        rows = _compute_and_print(title, windows, labels, model, scaler, ref_df, fn)
        all_rows.extend(rows)

    _run_blind_period_sim()

    # Summary
    import pandas as pd

    print(f"\n{'=' * _WIDTH}")
    print("  Summary: S(t) by Scenario (last window)")
    print(f"{'=' * _WIDTH}")
    summary = pd.DataFrame(all_rows)
    for scenario, sdf in summary.groupby("scenario", sort=False):
        last = sdf.iloc[-1]
        first_insuff = sdf[sdf["status"] == "insufficient"]
        cross_year = first_insuff.iloc[0]["year"] if len(first_insuff) > 0 else "never"
        print(f"  {scenario:40s}: S={last['S_t']:.3f}  insufficient_at={cross_year}")


if __name__ == "__main__":
    main()
