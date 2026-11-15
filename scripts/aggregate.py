"""Summarize results/*/seed*.json into results/summary.md.

Table 1: mean ± std over seeds. Table 2: test AUROC per test subject (mean over seeds),
because with 5 test subjects the between-subject spread dominates the seed spread.
"""
import glob
import json
from collections import defaultdict

import numpy as np

PAPER = "| paper, Table 3 | - | - | 0.6370 ± 0.0151 | 0.7139 ± 0.0088 | 0.6988 ± 0.0068 |"


def mean_std(values):
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def main():
    runs = defaultdict(list)
    for path in sorted(glob.glob("results/*/seed*.json")):
        with open(path) as f:
            run = json.load(f)
        runs[run["name"]].append(run)

    lines = [
        "| experiment | seeds | val AUROC | test BAcc | test AUC-PR | test AUROC |",
        "|---|---|---|---|---|---|",
        PAPER,
    ]
    for name, rs in runs.items():
        test = {k: mean_std([r["test"][k] for r in rs]) for k in ["bacc", "aucpr", "auroc"]}
        val = mean_std([r["val"]["auroc"] for r in rs])
        lines.append(f"| {name} | {len(rs)} | {val} | {test['bacc']} | {test['aucpr']} | {test['auroc']} |")

    subjects = sorted(next(iter(runs.values()))[0]["test_by_subject"], key=int)
    lines += [
        "",
        "Test AUROC per subject (mean over seeds)",
        "",
        "| experiment | " + " | ".join(f"sub {s}" for s in subjects) + " |",
        "|---|" + "---|" * len(subjects),
    ]
    for name, rs in runs.items():
        cells = [f"{np.mean([r['test_by_subject'][s]['auroc'] for r in rs]):.3f}" for s in subjects]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    summary = "\n".join(lines)
    print(summary)
    with open("results/summary.md", "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
