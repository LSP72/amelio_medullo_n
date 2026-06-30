import numpy as np


class ResultsCalculus:
    def __init__(self):
        pass

    @staticmethod
    def expected_calibration_error(y_true, y_proba, n_bins=5, strategy="quantile"):
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        if strategy == "quantile":
            edges = np.quantile(y_proba, np.linspace(0, 1, n_bins + 1))
            edges[0], edges[-1] = 0.0, 1.0  # ensure full coverage
            edges = np.unique(edges)  # guard against duplicate quantiles
        else:  # uniform
            edges = np.linspace(0, 1, n_bins + 1)

        # right-inclusive bins; last bin includes 1.0
        bin_ids = np.digitize(y_proba, edges[1:-1], right=True)

        ece = 0.0
        n = len(y_true)
        for b in range(len(edges) - 1):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            observed = y_true[mask].mean()
            predicted = y_proba[mask].mean()
            ece += (mask.sum() / n) * abs(observed - predicted)

        return ece
