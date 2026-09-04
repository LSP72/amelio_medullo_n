import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def split_into_blocks(group_sorted, block_sizes):
    """Split a patient's sorted sessions into consecutive blocks of the given sizes.

    Any leftover sessions past the last declared block size are kept as one
    final block, so a patient with 34 sessions and block_sizes=[20, 14] yields
    exactly two blocks, while block_sizes=[20] on the same patient yields a
    [20, 14] split too (the second block is inferred).
    """
    blocks = []
    start = 0
    for size in block_sizes:
        blocks.append(group_sorted.iloc[start : start + size])
        start += size
    if start < len(group_sorted):
        blocks.append(group_sorted.iloc[start:])
    return blocks


def plot_patient_regression(blocks, patient_id, feature, output_dir):
    """Fit and plot one regression line per block, all on the same figure."""
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 6))
    equation_lines = []
    block_results = []

    for i, (sessions, values) in enumerate(blocks):
        X = np.array(sessions).reshape(-1, 1)
        y = np.array(values)
        color = colors[i % len(colors)]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)

        label = f"Block {i + 1}" if len(blocks) > 1 else "Sessions"
        ax.scatter(X, y, color=color, alpha=0.6, label=label)
        ax.plot(X, y_pred, color=color)

        equation_lines.append(f"Block {i + 1}: y = {slope:.3f}x + {intercept:.3f}, R² = {r2:.3f}")
        block_results.append({"block": i + 1, "slope": slope, "intercept": intercept, "r2": r2})

    ax.text(
        0.05,
        0.95,
        "\n".join(equation_lines),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_title(f"Patient #{int(patient_id)} - {feature}")
    ax.set_xlabel("Session")
    ax.set_ylabel(feature)
    ax.legend()
    ax.grid(True, color="0.9")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, f"patient_{patient_id}_trend_for_{feature}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return block_results

def add_to_dict(ID, feature, block_results, results_dict):
    if ID not in results_dict:
        results_dict[ID] = {}
    results_dict[ID][feature] = block_results

def save_results_to_excel(results_dict, output_path):
    rows = []
    for ID, features in results_dict.items():
        for feature, block_results in features.items():
            for block in block_results:
                rows.append(
                    {
                        "ID": ID,
                        "Feature": feature,
                        "Block": block["block"],
                        "Slope": block["slope"],
                        "Intercept": block["intercept"],
                        "R2": block["r2"],
                    }
                )

    results_df = pd.DataFrame(rows)
    results_df.to_excel(output_path, index=False)

def plot_regressions_per_patient(
    data, feature_list, output_dir, id_col="ID", session_col="Session(s)", patient_blocks=None
):
    """
    patient_blocks: optional dict mapping a patient ID to a list of block sizes,
    e.g. {12345: [20, 16]} splits that patient's (sorted) sessions into a first
    block of 20 sessions and a second block of the remaining 16, each fit with
    its own regression line. Patients not listed get a single regression over
    all of their sessions.
    """
    patient_blocks = patient_blocks or {}
    results_dict = {}
    for feature in feature_list:
        for patient_id, group in data.groupby(id_col):
            group_sorted = group[[session_col, feature]].dropna().sort_values(session_col)

            if len(group_sorted) < 2:
                print(f"Skipping patient {patient_id}: not enough sessions with valid '{feature}' data.")
                continue

            block_sizes = patient_blocks.get(patient_id)
            chunks = split_into_blocks(group_sorted, block_sizes) if block_sizes else [group_sorted]

            blocks = []
            for i, chunk in enumerate(chunks):
                if len(chunk) < 2:
                    print(f"Skipping block {i + 1} for patient {patient_id} ({feature}): fewer than 2 sessions.")
                    continue
                blocks.append((chunk[session_col], chunk[feature]))

            if not blocks:
                continue

            block_results = plot_patient_regression(
                blocks=blocks,
                patient_id=patient_id,
                feature=feature,
                output_dir=output_dir,
            )

            add_to_dict(patient_id, feature, block_results, results_dict)

    save_results_to_excel(results_dict, os.path.join(output_dir, "patient_trends_summary.xlsx"))


if __name__ == "__main__":
    data = pd.read_excel(
        
    )

    feature_list = ["Vitesse_kmh_MOY", "BWS_%_MOY", "Guidage_G_%_MOY", "Guidage_D_%_MOY"]
    output_dir = "results/loko_results/patient_trends"

    # Patients whose sessions should be split into separate regression blocks,
    # e.g. before/after a change in protocol. Sizes are consecutive and in
    # session order; any leftover sessions form a final block automatically.
    patient_blocks = {
    
    }

    plot_regressions_per_patient(data, feature_list, output_dir, patient_blocks=patient_blocks)
