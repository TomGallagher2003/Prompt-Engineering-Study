
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Problem "steps" and binary difficulty mapping ----
PROBLEM_STEPS = {
    0: 4,   # Duck eggs -> subtract used -> leftovers -> revenue
    1: 3,   # Robe bolts -> compute white fiber -> add totals
    2: 4,   # House flip -> cost -> repairs -> new value -> profit
    3: 2,   # Sprints/week -> total meters
    4: 4,   # Chicken feed -> meals -> per chicken -> subtract given
    5: 4,   # Glasses -> cheap price -> pairs -> cost -> total
    6: 3,   # Sheep counts -> Charleston -> Toulouse -> sum
    7: 4,   # Download -> 40% time -> restart wait -> rest of file -> total
    8: 4,   # Trip -> outbound -> traffic -> 30 mph -> 80 mph
    9: 5,   # Pay -> base 40h -> rate -> overtime rate -> OT pay -> sum
    10: 3,  # Fractions -> simplify -> add
    11: 4,  # Cylinders -> volume per -> total volume -> convert units
    12: 4,  # Sale price -> fraction left -> multiply -> answer
    13: 3,  # Apples -> eat fraction -> remaining
    14: 4,  # Probability -> favorable outcomes -> total -> ratio
    15: 4,  # Speed/distance -> times -> compare -> answer
    16: 3,  # Area rectangle -> per unit -> multiply
    17: 4,  # Paint -> area -> coverage -> cans
    18: 2,  # Simple add/subtract
    19: 3,  # Workers -> hours each -> total
    20: 4,  # Mixture -> weights -> total
    21: 3,  # Division into equal parts
    22: 3,  # Travel -> speed × time
    23: 5,  # Interest -> principal -> rate -> time -> total
    24: 3,  # Discounted price -> percent relation -> solve
    25: 4,  # Ratios -> convert -> solve
    26: 4,  # Probability compound -> multiply -> result
    27: 4,  # Work problem -> rate addition -> time
    28: 3,  # Temperature convert -> formula
    29: 4,  # Average speed multi-leg -> distances -> times -> total
    30: 5,  # Compound interest -> compute growth
    31: 4,  # Area composite -> break shapes -> add
    32: 3,  # Perimeter -> sides -> sum
    33: 3,  # Time to fill/drain -> rates -> net
    34: 4,  # Ratio sharing -> compute each share
    35: 5,  # Profit/loss % -> initial vs final
    36: 5,  # Work mixture -> portions -> combine -> total
    37: 4,  # Speed/time/distance -> multi-step
    38: 3,  # Simple linear equation solve
    39: 3,  # Compare quantities
    40: 4,  # Geometry volume -> formula -> calc
    41: 5,  # Compound probability
    42: 4,  # Work/backtracking -> time calc
    43: 5,  # Geometry area composite
    44: 4,  # Percentage discount then add
    45: 4,  # Ratio/proportion solve
    46: 5,  # Train problem -> relative speed
    47: 4,  # Geometry perimeter
    48: 3,  # Money exchange simple
    49: 4,  # Profit distribution
    50: 3,  # Simple average
    51: 3,  # Speed = dist/time
    52: 5,  # Work efficiency multi-worker
    53: 5,  # Probability tree
    54: 4,  # Compound % discount
    55: 3,  # Simple mixture
    56: 4,  # Simple interest
    57: 5,  # Geometry volume composite
    58: 5,  # Distance/speed/time multiple stages
    59: 4,  # Probability conditional
    60: 4,  # Percent increase then decrease
    61: 5,  # Work-time distribution
    62: 3,  # Perimeter basic
    63: 4,  # Ratio division
    64: 5,  # Geometry triangle area + ratio
    65: 3,  # Travel distance
    66: 4,  # Simple algebra eqn
    67: 3,  # Average marks
    68: 4,  # Compound % increase
    69: 4,  # Interest over years
}

# Label rule: <5 steps => easy, else hard
PROBLEM_DIFFICULTY_LABELS: Dict[int, str] = {i: ("easy" if steps < 4 else "hard") for i, steps in PROBLEM_STEPS.items()}


def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "correct" in df.columns:
        df["correct"] = (
            df["correct"].astype(str).str.strip().str.lower()
            .map({"1": 1, "0": 0, "true": 1, "false": 0})
            .fillna(df["correct"]).astype(float).fillna(0).astype(int)
        )
    for col in ("gold", "pred"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for alias in ("strategy", "prompt_type"):
        if alias in df.columns and "technique" not in df.columns:
            df.rename(columns={alias: "technique"}, inplace=True)
    if "technique" not in df.columns:
        df["technique"] = "unknown"
    return df


def accuracy_by_technique(df: pd.DataFrame) -> pd.Series:
    if "correct" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("technique")["correct"].mean().sort_values(ascending=False)


def accuracy_by_technique_and_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    if "correct" not in df.columns:
        return pd.DataFrame()
    if "idx" not in df.columns:
        raise ValueError("Expected 'idx' column to map problem difficulties.")
    df = df.copy()
    # Map idx -> 'easy'/'hard' labels (IMPORTANT: not numeric steps)
    df["difficulty"] = df["idx"].map(PROBLEM_DIFFICULTY_LABELS)
    df = df[~df["difficulty"].isna()]
    piv = (df
           .groupby(["difficulty", "technique"])["correct"]
           .mean()
           .unstack("technique"))
    # Force difficulty order
    difficulty_order = ["easy", "hard"]
    piv = piv.reindex(index=[d for d in difficulty_order if d in piv.index])
    return piv


def save_table(series_or_df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    series_or_df.to_csv(path)


def plot_accuracy_by_technique(series: pd.Series, out: Path):
    fig = plt.figure()
    series.plot(kind="bar", rot=45)
    plt.title("Accuracy by Technique")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_grouped_by_difficulty(df: pd.DataFrame, out: Path, technique_order: List[str]):
    if df.empty:
        return
    plot_df = df.T  # techniques as index
    plot_df = plot_df.reindex(index=[t for t in technique_order if t in plot_df.index])
    fig = plt.figure()
    ax = plt.gca()
    n_groups = plot_df.shape[0]
    n_series = plot_df.shape[1]
    idx = np.arange(n_groups)
    width = 0.8 / max(n_series, 1)
    for i, col in enumerate(plot_df.columns):
        ax.bar(idx + i*width, plot_df[col].values, width=width, label=str(col))
    ax.set_xticks(idx + (n_series-1)*width/2)
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Technique across Difficulty")
    ax.legend(title="Difficulty")
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Analyze results with binary difficulty (easy/hard).")
    ap.add_argument("csv", type=Path, help="Path to results.csv")
    ap.add_argument("--outdir", type=Path, default=Path("analysis_out_min_easyhard"), help="Output directory")
    args = ap.parse_args()

    df = safe_read_csv(args.csv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    by_tech = accuracy_by_technique(df)
    save_table(by_tech.rename("accuracy"), outdir / "accuracy_by_technique.csv")
    plot_accuracy_by_technique(by_tech, outdir / "accuracy_by_technique.png")

    by_diff = accuracy_by_technique_and_difficulty(df)
    save_table(by_diff, outdir / "accuracy_by_technique_by_easy_hard.csv")
    plot_grouped_by_difficulty(by_diff, outdir / "accuracy_by_technique_by_easy_hard.png", technique_order=list(by_tech.index))

    print("Saved:")
    print(f" - {outdir/'accuracy_by_technique.csv'}")
    print(f" - {outdir/'accuracy_by_technique.png'}")
    print(f" - {outdir/'accuracy_by_technique_by_easy_hard.csv'}")
    print(f" - {outdir/'accuracy_by_technique_by_easy_hard.png'}")


if __name__ == "__main__":
    main()
