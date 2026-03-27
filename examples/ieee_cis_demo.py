"""IEEE-CIS Fraud Detection — Evidence Sufficiency Calculator evaluation.

Computes S(t) trajectories on real IEEE-CIS data across temporal windows
with three drift injection scenarios. Demonstrates blind period simulation,
threshold crossings, and governance response classification.

This demo complements the the Governance Drift Toolkit monitoring demo. The drift toolkit detects drift;
the sufficiency calculator quantifies its governance impact via sufficiency scoring.

Prerequisites
-------------
1. Install demo dependencies::

       pip install -e ".[demo]"

2. Clone the separate Governance Drift Toolkit repository next to this one.

3. Prepare the IEEE-CIS demo data in that repository. The raw dataset comes
   from Kaggle and is documented there in its own demo instructions.

4. This example expects the CSV at::

       ../governance-drift-toolkit/data/ieee_cis/train_transaction.csv

   A typical local setup is::

    cd ../governance-drift-toolkit
    make demo-ieee

5. Run::

       python examples/ieee_cis_demo.py
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

DATA_PATH = Path("../governance-drift-toolkit/data/ieee_cis/train_transaction.csv")
WINDOW_DAYS = 30
SECONDS_PER_DAY = 86_400
SEED = 42

FEATURE_COLS = [
    "TransactionAmt",
    "card1",
    "addr1",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V12",
    "V13",
    "V14",
    "V15",
    "V29",
    "V30",
    "V33",
    "V34",
    "V44",
    "V45",
    "V46",
    "V47",
    "V48",
    "V54",
    "V55",
    "V56",
    "V57",
    "V69",
    "V70",
    "V71",
    "V72",
    "V73",
    "V74",
    "V75",
    "V76",
    "V77",
    "V78",
    "V83",
    "V87",
    "V126",
    "V127",
    "V128",
    "V129",
    "V130",
    "V131",
    "V279",
    "V280",
    "V282",
    "V283",
    "V306",
    "V307",
    "V308",
    "V309",
    "V310",
    "V312",
    "V313",
    "V314",
    "V315",
]

# IEEE-CIS specific: tau_r calibrated for 3.5% fraud imbalance.
# F1 ~ 0.13 is the realistic baseline for logistic regression on this data;
# tau_r = 0.15 places the gate threshold just above baseline so that
# degradation produces visible S(t) dynamics.
IEEE_CIS_TAU_R = 0.15

SHIFT_FEATURES = ["TransactionAmt", "V1", "V3"]
COVARIATE_SIGMAS = [0.3, 0.6, 1.0, 1.5, 2.0]
CONCEPT_FLIP_RATES = [0.10, 0.25, 0.50, 0.75, 0.95]  # fraud labels flipped to legit
MIXED_SIGMAS = [0.2, 0.4, 0.7, 1.0, 1.5]
MIXED_FLIPS = [0.05, 0.15, 0.30, 0.50, 0.70]  # fraud labels flipped to legit


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _ieee_cis_config():
    """Fraud detection config with tau_r calibrated for IEEE-CIS class imbalance."""
    from sufficiency._dimensions import COMPLETENESS, FRESHNESS, RELIABILITY, REPRESENTATIVENESS
    from sufficiency.types import GovernanceConfig, SufficiencyThresholds

    return GovernanceConfig(
        weights={
            COMPLETENESS: 0.20,
            FRESHNESS: 0.30,
            RELIABILITY: 0.30,
            REPRESENTATIVENESS: 0.20,
        },
        tau_c=0.6,
        tau_r=IEEE_CIS_TAU_R,
        lambda_freshness=0.02,
        ks_cap=0.30,
        thresholds=SufficiencyThresholds(sufficient=0.8, degraded=0.5),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_and_split() -> list[pd.DataFrame]:
    """Load IEEE-CIS data and split into 30-day windows."""
    import pandas as pd

    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        print("This example expects a sibling checkout of governance-drift-toolkit.")
        print("Prepare the dataset there, then rerun this example:")
        print("  cd ../governance-drift-toolkit && make demo-ieee")
        sys.exit(1)

    print("Loading IEEE-CIS train_transaction.csv ...")
    df = pd.read_csv(DATA_PATH, usecols=["TransactionDT", "isFraud", *FEATURE_COLS])
    df["day"] = (df["TransactionDT"] - df["TransactionDT"].min()) // SECONDS_PER_DAY

    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    total_days = int(df["day"].max()) + 1
    n_win = total_days // WINDOW_DAYS
    windows = []
    for w in range(n_win):
        s, e = w * WINDOW_DAYS, (w + 1) * WINDOW_DAYS
        wdf = df[(df["day"] >= s) & (df["day"] < e)].copy()
        if len(wdf) > 0:
            windows.append(wdf)

    print(f"  {len(df):,} transactions, {total_days} days, {len(windows)} windows")
    return windows


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
    """One-directional flip: fraud→legit only, so R(t) decreases."""
    out = df.copy()
    flip_rate = CONCEPT_FLIP_RATES[min(idx, len(CONCEPT_FLIP_RATES) - 1)]
    fraud_idx = out[out["isFraud"] == 1].index
    n_flip = min(int(len(fraud_idx) * flip_rate), len(fraud_idx))
    flip_idx = rng.choice(fraud_idx, size=n_flip, replace=False)
    out.loc[flip_idx, "isFraud"] = 0
    return out


def _inject_mixed(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> pd.DataFrame:
    """Covariate shift + one-directional fraud→legit label flip."""
    sigma = MIXED_SIGMAS[min(idx, len(MIXED_SIGMAS) - 1)]
    flip_rate = MIXED_FLIPS[min(idx, len(MIXED_FLIPS) - 1)]
    out = df.copy()
    for col in SHIFT_FEATURES:
        out[col] = out[col] + rng.normal(0, sigma * out[col].std(), len(out))
    fraud_idx = out[out["isFraud"] == 1].index
    n_flip = min(int(len(fraud_idx) * flip_rate), len(fraud_idx))
    flip_idx = rng.choice(fraud_idx, size=n_flip, replace=False)
    out.loc[flip_idx, "isFraud"] = 0
    return out


# ---------------------------------------------------------------------------
# Sufficiency computation per window
# ---------------------------------------------------------------------------


def _compute_window_sufficiency(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    model: object,
    scaler: object,
    window_day: int,
) -> dict:
    """Compute S(t) for one window using sufficiency dimension scorers."""
    from sklearn.metrics import f1_score as sk_f1

    from sufficiency import compute_sufficiency
    from sufficiency.dimensions.completeness import compute_completeness
    from sufficiency.dimensions.freshness import compute_freshness
    from sufficiency.dimensions.reliability import compute_reliability
    from sufficiency.dimensions.representativeness import compute_representativeness

    config = _ieee_cis_config()

    # Completeness: fraction of transactions with known isFraud
    # In blind period simulation, assume labels arrive for a decreasing fraction
    label_availability = max(0.3, 1.0 - window_day * 0.004)
    n_total = len(cur_df)
    n_labeled = int(n_total * label_availability)

    # Freshness: days since reference window
    delta_t = float(window_day)

    # Reliability: F1 on labeled subset
    y_true = cur_df["isFraud"].values[:n_labeled]
    x_cur = scaler.transform(cur_df[FEATURE_COLS].values[:n_labeled])  # type: ignore[union-attr]
    y_pred = (model.predict_proba(x_cur)[:, 1] > 0.5).astype(int)  # type: ignore[union-attr]
    actual_f1 = float(sk_f1(y_true, y_pred)) if len(y_true) > 10 else 0.5

    # Representativeness: KS between ref and current score distributions
    ref_scores = model.predict_proba(  # type: ignore[union-attr]
        scaler.transform(ref_df[FEATURE_COLS].values)  # type: ignore[union-attr]
    )[:, 1]
    cur_scores = model.predict_proba(  # type: ignore[union-attr]
        scaler.transform(cur_df[FEATURE_COLS].values)  # type: ignore[union-attr]
    )[:, 1]

    dimensions = {
        "completeness": compute_completeness(n_labeled, n_total),
        "freshness": compute_freshness(delta_t, config.lambda_freshness),
        "reliability": compute_reliability(y_true, y_pred, rng_seed=SEED),
        "representativeness": compute_representativeness(ref_scores, cur_scores),
    }

    result = compute_sufficiency(dimensions, config)

    return {
        "day": window_day,
        "n_txns": n_total,
        "fraud_rate": cur_df["isFraud"].mean(),
        "S_t": result.composite,
        "gate": result.gate,
        "status": result.status.value,
        "C": dimensions["completeness"].value,
        "F": dimensions["freshness"].value,
        "R": dimensions["reliability"].value,
        "P": dimensions["representativeness"].value,
        "actual_f1": actual_f1,
        "label_avail": label_availability,
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

_WIDTH = 110


def _run_scenario(
    title: str,
    windows: list[pd.DataFrame],
    model: object,
    scaler: object,
    inject_fn: object | None = None,
) -> list[dict]:
    """Run sufficiency scoring across windows with optional drift injection."""
    rng = np.random.default_rng(SEED)
    ref_df = windows[0]

    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")
    print(
        f"{'Win':>3} | {'Day':>5} | {'Txns':>7} | {'Fraud%':>6} | "
        f"{'C':>5} | {'F':>5} | {'R':>5} | {'P':>5} | "
        f"{'A(t)':>5} | {'S(t)':>5} | {'Status':>12} | {'F1':>5}"
    )
    print(f"{'-' * _WIDTH}")

    rows = []
    for i, raw_df in enumerate(windows[1:]):
        current_df = inject_fn(raw_df, i, rng) if inject_fn is not None else raw_df
        window_day = (i + 1) * WINDOW_DAYS
        m = _compute_window_sufficiency(ref_df, current_df, model, scaler, window_day)
        m["window"] = i + 1
        m["scenario"] = title
        rows.append(m)

        print(
            f"{i + 1:>3} | {window_day:>5} | {m['n_txns']:>7,} | {m['fraud_rate']:>5.3f} | "
            f"{m['C']:>5.3f} | {m['F']:>5.3f} | {m['R']:>5.3f} | {m['P']:>5.3f} | "
            f"{m['gate']:>5.3f} | {m['S_t']:>5.3f} | {m['status']:>12} | {m['actual_f1']:>5.3f}"
        )

    return rows


# ---------------------------------------------------------------------------
# Blind period simulation comparison
# ---------------------------------------------------------------------------


def _run_blind_period_simulation(
    empirical_c: float,
    empirical_r: float,
    empirical_p: float,
) -> None:
    """Run blind period simulator calibrated from empirical reference window."""
    from sufficiency import BlindPeriodSimulator, DriftSpec, DriftType

    config = _ieee_cis_config()

    print(f"\n{'=' * _WIDTH}")
    print(
        f"  Blind Period Simulation (calibrated: C={empirical_c:.3f}, "
        f"R={empirical_r:.3f}, P={empirical_p:.3f})"
    )
    print(f"{'=' * _WIDTH}")

    for drift_name, specs in [
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
            initial_completeness=empirical_c,
            initial_reliability=empirical_r,
            initial_representativeness=empirical_p,
            config=config,
            drift_specs=specs,
        )

        print(f"\n  {drift_name}:")
        for r in sim.simulate([30, 60, 90, 180]):
            days = (r.timestamp - sim.start_time).days
            print(f"    Day {days:3d}: S={r.composite:.3f}  A={r.gate:.3f}  {r.status.value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run sufficiency evaluation on IEEE-CIS data."""
    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Missing dependencies. Install with: pip install -e '.[demo]'")
        sys.exit(1)

    windows = _load_and_split()

    # Train reference model
    ref_df = windows[0]
    scaler = StandardScaler()
    x_ref = scaler.fit_transform(ref_df[FEATURE_COLS].values)
    y_ref = ref_df["isFraud"].values
    model = LogisticRegression(
        max_iter=1000, random_state=SEED, solver="lbfgs", class_weight="balanced"
    )
    model.fit(x_ref, y_ref)

    # Compute empirical reference dimensions for simulator calibration
    from sklearn.metrics import f1_score as sk_f1

    from sufficiency.dimensions.completeness import compute_completeness
    from sufficiency.dimensions.reliability import compute_reliability
    from sufficiency.dimensions.representativeness import compute_representativeness

    ref_probs = model.predict_proba(x_ref)[:, 1]
    ref_preds = (ref_probs > 0.5).astype(int)
    ref_f1 = float(sk_f1(y_ref, ref_preds))
    ref_c = compute_completeness(len(y_ref), len(ref_df)).value
    ref_r = compute_reliability(y_ref, ref_preds, rng_seed=SEED).value
    ref_p = compute_representativeness(ref_probs, ref_probs).value  # self-comparison = 1.0
    print(f"  Reference: {len(ref_df):,} txns, fraud_rate={y_ref.mean():.4f}, F1={ref_f1:.3f}")
    print(f"  Empirical dimensions: C={ref_c:.3f}, R={ref_r:.3f}, P={ref_p:.3f}")

    # Scenario 1-4: Real data + injection
    all_rows: list[dict] = []
    for title, fn in [
        ("Baseline (no drift)", None),
        ("Covariate Drift P(X)", _inject_covariate),
        ("Mixed Drift P(X)+P(Y|X)", _inject_mixed),
        ("Pure Concept Drift P(Y|X)", _inject_concept),
    ]:
        rows = _run_scenario(title, windows, model, scaler, inject_fn=fn)
        all_rows.extend(rows)

    # Blind period simulation calibrated from empirical reference
    _run_blind_period_simulation(ref_c, ref_r, ref_p)

    # Summary
    print(f"\n{'=' * _WIDTH}")
    print("  Summary: Sufficiency Degradation by Structural Condition")
    print(f"{'=' * _WIDTH}")
    import pandas as pd

    summary = pd.DataFrame(all_rows)
    for scenario, sdf in summary.groupby("scenario", sort=False):
        last = sdf.iloc[-1]
        crossed = (sdf["S_t"] < 0.8).any()
        cross_day = int(sdf.loc[sdf["S_t"] < 0.8, "day"].min()) if crossed else -1
        print(
            f"  {scenario:30s}: final S={last['S_t']:.3f} "
            f"threshold_crossed={'day ' + str(cross_day) if crossed else 'never':>8}"
        )


if __name__ == "__main__":
    main()
