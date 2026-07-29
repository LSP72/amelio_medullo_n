"""
Permutation test for the feature-combination search.

WHAT IT MEASURES
----------------
Your reported 0.7936 is the MAXIMUM AUC over ~7,800 feature combinations.
A maximum over many noisy estimates is biased upward even when nothing is real.
This test asks: how large a max-AUC does your ENTIRE pipeline produce when the
labels carry no information? We destroy the label-feature link by shuffling y,
then run the whole search (all combinations x all splits x max-selection) and
record the single best AUC. Repeat many times -> null distribution of the max.

Your real 0.7936 is then compared against that null. If it sits above the 99th
percentile, the search found signal that survives the multiplicity. If it's in
the bulk, your winner is what searching 7,800 combinations on 92 people buys you
by chance.

WHY THIS IS VALID
-----------------
- We import your own functions. The per-combination evaluation is byte-identical
  to your real run: same split seeds, same CatBoost config, same in-fold cat fill.
- y is shuffled ONCE per permutation, AFTER MCID labels are computed, and BEFORE
  the combination loop. The split structure is held fixed; only the label-feature
  association is broken. That is the definition of this null.
- The statistic recorded per permutation is the MAX over all combinations, not
  any single set's AUC. That is the whole point.

COST
----
Full real search at 20 splits x sizes 4-13 is ~7,814 combos x 20 fits per combo.
Times N_PERM permutations. Use the knobs below to keep laptop runtime in hours.
Start SMALL (N_PERM=20, N_SPLITS=10, sizes 4-8) to confirm it runs and to time
one permutation, THEN scale up. One permutation's wall-time x N_PERM is your budget.
"""

import os
import time
import numpy as np
import pandas as pd
from itertools import combinations

# --- import YOUR pipeline unchanged -----------------------------------------
# Adjust this import to point at the module holding your original script.
# If your file is `feature_search.py`, this is `from feature_search import ...`.
from catboost_feature_combo import (
    train_single_split,      # reused verbatim -> identical per-fold behavior
    DataCleaning,
    Calculus,
)

# ============================ CONFIG =========================================
DATA_PATH = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/final_data_matrix_sessions_separated.xlsx"

ALL_COLS = [
    "Neurol_cond", "Lesion_num", "Nb sessions", "Sex", "Age", "BMI",
    "6MWT_m_pre", "10MWT_pas_pre", "10MWT_sec_pre", "delay_injury",
    "delay_loko", "functional_level", "speed",
]

# --- knobs: start small, scale after timing one permutation ------------------
N_PERM       = 20                      # permutations. 200-300 for a real tail estimate.
N_SPLITS     = 10                      # splits per combo. Fewer = faster, noisier per-combo.
MIN_FEATURES = 4
MAX_FEATURES = 8                       # 9-13 are already dead; scoping to 4-8 is defensible.
SPLIT_SEEDS  = np.arange(1, N_SPLITS + 1)   # SAME seeds every permutation -> split structure fixed
REAL_BEST_AUC = 0.7936                 # your observed max, for the final comparison
OUT_DIR      = "results/permutation_null"
NUM          = False                   # matches your real run's `num=False`
# =============================================================================


def load_Xy():
    """Reproduce your real data prep exactly (from search_all_feature_combinations)."""
    data = pd.read_excel(DATA_PATH)
    if NUM:
        data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
        data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)
    X = data[ALL_COLS]
    y = Calculus.calculate_MCID_2(data, default_threshold=45)["MCID_classes"]
    # reset index so positional shuffling of y aligns with X rows
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    return X, y


def run_full_search_max(X, y):
    """
    Run the ENTIRE search for one (already-prepared) X, y and return the single
    best mean-AUC across all combinations. Mirrors your real loop but only tracks
    the max (we don't need the full table for the null).
    """
    best = -np.inf
    for size in range(MIN_FEATURES, MAX_FEATURES + 1):
        for combo in combinations(ALL_COLS, size):
            X_sub = X[list(combo)]
            aucs = [train_single_split(X_sub, y, s)[0] for s in SPLIT_SEEDS]
            m = float(np.mean(aucs))
            if m > best:
                best = m
    return best


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = load_Xy()
    y_arr = y.to_numpy()
    n_combos = sum(len(list(combinations(ALL_COLS, s)))
                   for s in range(MIN_FEATURES, MAX_FEATURES + 1))
    print(f"Combinations per permutation: {n_combos} | splits each: {N_SPLITS}")
    print(f"Fits per permutation: ~{n_combos * N_SPLITS:,} | permutations: {N_PERM}\n")

    rng = np.random.default_rng(0)   # controls the label shuffles, reproducibly
    null_maxes = []

    for p in range(N_PERM):
        t0 = time.time()
        # ---- the ONE shuffle: break label-feature link, once, before the search
        y_perm = pd.Series(rng.permutation(y_arr))
        best = run_full_search_max(X, y_perm)
        null_maxes.append(best)
        dt = time.time() - t0
        print(f"perm {p+1:3d}/{N_PERM}  null max AUC = {best:.4f}  ({dt:.1f}s)"
              + ("   <-- first perm: multiply by N_PERM for total budget" if p == 0 else ""))
        pd.Series(null_maxes).to_csv(f"{OUT_DIR}/null_max_aucs.csv", index=False)

    null = np.array(null_maxes)
    pctl = {q: float(np.percentile(null, q)) for q in (50, 90, 95, 99)}
    p_value = (np.sum(null >= REAL_BEST_AUC) + 1) / (len(null) + 1)  # +1: never report p=0

    print("\n================ RESULT ================")
    print(f"Null max-AUC:  median {pctl[50]:.4f} | 90th {pctl[90]:.4f} "
          f"| 95th {pctl[95]:.4f} | 99th {pctl[99]:.4f}")
    print(f"Your real max-AUC: {REAL_BEST_AUC:.4f}")
    print(f"Permutation p-value: {p_value:.4f}  (from {len(null)} permutations)")
    if REAL_BEST_AUC >= pctl[99]:
        print("-> Above the 99th percentile of chance: signal survives the search.")
    elif REAL_BEST_AUC <= pctl[90]:
        print("-> Inside the null bulk: the winner is consistent with chance.")
    else:
        print("-> Borderline. This many permutations can't separate real from edge-of-chance;")
        print("   the tail estimate is too noisy here. Interpret as 'weak at best'.")
    print("========================================")


if __name__ == "__main__":
    main()