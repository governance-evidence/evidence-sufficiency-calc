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
    "V12",
    "V13",
    "V14",
    "V54",
    "V75",
    "V78",
]

SHIFT_FEATURES = ["TransactionAmt", "V1", "V3"]
COVARIATE_SIGMAS = [0.3, 0.6, 1.0, 1.5, 2.0]
CONCEPT_FLIP_RATES = [0.05, 0.10, 0.20, 0.35, 0.50]
MIXED_SIGMAS = [0.2, 0.4, 0.7, 1.0, 1.5]
MIXED_FLIPS = [0.03, 0.07, 0.12, 0.20, 0.30]


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
    out = df.copy()
    flip_rate = CONCEPT_FLIP_RATES[min(idx, len(CONCEPT_FLIP_RATES) - 1)]
    n_flip = int(len(out) * flip_rate)
    flip_idx = rng.choice(out.index, size=n_flip, replace=False)
    out.loc[flip_idx, "isFraud"] = 1 - out.loc[flip_idx, "isFraud"]
    return out


def _inject_mixed(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> pd.DataFrame:
    sigma = MIXED_SIGMAS[min(idx, len(MIXED_SIGMAS) - 1)]
    flip_rate = MIXED_FLIPS[min(idx, len(MIXED_FLIPS) - 1)]
    out = df.copy()
    for col in SHIFT_FEATURES:
        out[col] = out[col] + rng.normal(0, sigma * out[col].std(), len(out))
    n_flip = int(len(out) * flip_rate)
    flip_idx = rng.choice(out.index, size=n_flip, replace=False)
    out.loc[flip_idx, "isFraud"] = 1 - out.loc[flip_idx, "isFraud"]
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

    from sufficiency import compute_sufficiency, fraud_detection_config
    from sufficiency.dimensions.completeness import compute_completeness
    from sufficiency.dimensions.freshness import compute_freshness
    from sufficiency.dimensions.reliability import compute_reliability
    from sufficiency.dimensions.representativeness import compute_representativeness

    config = fraud_detection_config()

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


def _run_blind_period_simulation() -> None:
    """Run blind period simulator with three drift types."""
    from sufficiency import BlindPeriodSimulator, DriftSpec, DriftType, fraud_detection_config

    config = fraud_detection_config()

    print(f"\n{'=' * _WIDTH}")
    print("  Blind Period Simulation (BlindPeriodSimulator)")
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
            initial_completeness=0.85,
            initial_reliability=0.88,
            initial_representativeness=0.95,
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
    except ImportError:
        print("Missing dependencies. Install with: pip install -e '.[demo]'")
        sys.exit(1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    windows = _load_and_split()

    # Train reference model
    ref_df = windows[0]
    scaler = StandardScaler()
    x_ref = scaler.fit_transform(ref_df[FEATURE_COLS].values)
    y_ref = ref_df["isFraud"].values
    model = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")
    model.fit(x_ref, y_ref)
    print(f"  Reference: {len(ref_df):,} txns, fraud_rate={y_ref.mean():.4f}")

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

    # Blind period simulation (analytic, no data needed)
    _run_blind_period_simulation()

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
