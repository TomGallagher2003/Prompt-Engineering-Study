#!/usr/bin/env python3
"""
Compute and plot percentage accuracy decrease from easy → hard for each technique.

Usage:
    python plot_easy_hard_drop.py --csv analysis_out_min_easyhard/accuracy_by_technique_by_easy_hard.csv --out plot.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to accuracy_by_technique_by_easy_hard.csv")
    parser.add_argument("--out", default="easy_hard_drop.png", help="Output PNG filename")
    args = parser.parse_args()

    # Load pivot table (index = easy/hard, columns = techniques)
    df = pd.read_csv(args.csv, index_col=0)

    drops = {}
    for tech in df.columns:
        if "easy" in df.index and "hard" in df.index:
            easy = df.loc["easy", tech]
            hard = df.loc["hard", tech]
            if easy > 0:
                drops[tech] = max((easy - hard) / easy * 100, 0)

    drop_df = pd.DataFrame.from_dict(drops, orient="index", columns=["% decrease"]).sort_values("% decrease", ascending=False)
    print(drop_df)

    # Plot
    ax = drop_df.plot(kind="bar", legend=False)
    ax.set_ylabel("% Accuracy Decrease (Easy → Hard)")
    ax.set_title("Accuracy Drop by Technique")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved plot → {args.out}")


if __name__ == "__main__":
    main()
