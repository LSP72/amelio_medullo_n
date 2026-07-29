import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from amelio_medullo import DataCleaning, Calculus

# --- Déclare explicitement le type de chaque variable ---
NOMINAL_COLS = ["Neurol_cond", "Sex"]  # ajoute functional_level / Lesion_num si nominales
# Tout le reste de cols_to_keep sera traité comme numérique/continu.


def load_data(data_path, cols_to_keep):
    data = pd.read_excel(data_path)

    # Recode lisible AVANT le one-hot (les labels deviennent des noms de colonnes parlants)
    data["Neurol_cond"] = data["Neurol_cond"].map({"BM": "BM", "AVC": "AVC", "Autre": "Autre"})
    data["Sex"] = data["Sex"].map({"M": "M", "F": "F"})

    # /!\ apply NE modifie PAS en place : on réassigne (corrige ton bug n°2)
    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)

    # Cible binaire
    y = Calculus.calculate_MCID_2(data, 45)["MCID_classes"].rename("MCID")

    # Copie explicite pour éviter le SettingWithCopyWarning
    X = data[cols_to_keep].copy()

    # On aligne X et y, puis on droppe les NaN sur l'ensemble d'un coup
    df = X.join(y)
    df = df.dropna(axis=0)

    y_clean = df["MCID"].astype(int)
    X_clean = df.drop(columns=["MCID"])
    return X_clean, y_clean


def build_design_matrix(X, nominal_cols, scale=True):
    """One-hot des nominales + standardisation des continues."""
    nominal_present = [c for c in nominal_cols if c in X.columns]
    continuous = [c for c in X.columns if c not in nominal_present]

    X_num = X[continuous].astype(float).copy()
    if scale:
        scaler = StandardScaler()
        X_num[continuous] = scaler.fit_transform(X_num[continuous])
        # NB: fit sur tout X = OK pour inférence; pour de la prédiction, fit sur le train seulement.

    X_cat = pd.get_dummies(X[nominal_present].astype("category"), drop_first=True, dtype=float)

    X_design = pd.concat([X_num, X_cat], axis=1)
    return X_design


# ----------------- UNIVARIÉ -----------------
def univariate_screen(X, y, nominal_cols):
    """Une régression logistique par variable. Renvoie OR (par écart-type pour les continues), IC, p."""
    rows = []
    for col in X.columns:
        xi = X[[col]]
        if col in nominal_cols:
            xi = pd.get_dummies(xi.astype("category"), drop_first=True, dtype=float)
        else:
            xi = pd.DataFrame(
                StandardScaler().fit_transform(xi.astype(float)), columns=[col], index=xi.index
            )  # OR "par SD"
        xi = sm.add_constant(xi)
        try:
            res = sm.Logit(y, xi).fit(disp=0)
            for name in xi.columns:
                if name == "const":
                    continue
                rows.append(
                    {
                        "variable": name,
                        "OR": np.exp(res.params[name]),
                        "CI_low": np.exp(res.conf_int().loc[name, 0]),
                        "CI_high": np.exp(res.conf_int().loc[name, 1]),
                        "p_value": res.pvalues[name],
                    }
                )
        except Exception as e:
            rows.append(
                {"variable": col, "OR": np.nan, "CI_low": np.nan, "CI_high": np.nan, "p_value": np.nan, "error": str(e)}
            )
    out = pd.DataFrame(rows).sort_values("p_value")
    return out


# ----------------- MULTIVARIÉ -----------------
def multivariate_logit(X, y, nominal_cols, scale=True):
    X_design = build_design_matrix(X, nominal_cols, scale=scale)
    X_design = sm.add_constant(X_design)
    model = sm.Logit(y, X_design).fit()
    print(model.summary())

    # Odds ratios + IC, plus lisibles que les coefficients bruts
    or_table = pd.DataFrame(
        {
            "OR": np.exp(model.params),
            "CI_low": np.exp(model.conf_int()[0]),
            "CI_high": np.exp(model.conf_int()[1]),
            "p_value": model.pvalues,
        }
    )
    print("\nOdds ratios:\n", or_table)
    return model, or_table


def main(data_path, top_features):
    X, y = load_data(data_path, top_features)
    print(f"n = {len(y)}, répondeurs = {int(y.sum())}, non-répondeurs = {int((1 - y).sum())}")

    # 1) Screening univarié pour voir le terrain
    uni = univariate_screen(X, y, NOMINAL_COLS)
    print(uni)

    # 2) Modèle multivarié (réduis top_features AVANT d'arriver ici)
    model, or_table = multivariate_logit(X, y, NOMINAL_COLS, scale=True)
    return model, or_table


if __name__ == "__main__":
    data_path = "/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/CHUNantes collaboration/donnees/data_from_dpi/merged_data_final.xlsx"

    # SELECTED FEATURES
    cols_to_keep = ["step_length", "Durée_min", "BWS_%_MOY", "duration", "Neurol_cond", "sessions_per_week"]
    # All features with high correlated removed (i.e., cadence, speed)
    # cols_to_keep_2 = [
    #     "nb_sessions",
    #     # "duration",
    #     "Durée_min",
    #     "Vitesse_kmh_MOY",
    #     "BWS_%_MOY",
    #     "step_length",
    #     "Guidage_%_MOY",
    #     "sessions_per_week",
    #     "Neurol_cond",
    #     "Sex",
    #     "Age",
    #     "Nb sessions",
    #     "functional_level",
    #     "Lesion_num",
    #     "BMI",
    # ]

    main(data_path, cols_to_keep)
