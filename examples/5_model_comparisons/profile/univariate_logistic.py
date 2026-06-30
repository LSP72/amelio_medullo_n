"""
Screening univarié de biomarqueurs vs cible binaire MCID.

Corrections par rapport à la version initiale :
  (1) data.apply(...) est désormais RÉASSIGNÉ (le nettoyage n'était jamais conservé).
  (2) Les variables NOMINALES (ex. Neurol_cond) sont one-hot encodées et testées par un
      test global du rapport de vraisemblance (k-1 ddl), pas par un Wald à 1 ddl sur un
      code 1/2/3 arbitraire.
  (3) L'AUC est explicitement nommée "AUC_apparent" : elle est calculée IN-SAMPLE, donc
      optimiste. Utilisable pour CLASSER, pas pour annoncer une capacité de discrimination.
  (4) dropna PAR PAIRE (feature testée + cible) dans chaque test, au lieu d'un dropna
      global sur les 12 colonnes qui amputait chaque test à cause des NaN des AUTRES.
  (5) Plus de suffixe "_std" trompeur : on fournit explicitement OR_per_unit ET OR_per_SD.
  Mineurs : except Exception as e (plus de except nu), création du dossier de sortie,
            colonne 'n' = taille d'échantillon réellement utilisée par chaque test.
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from amelio_medullo import DataCleaning, Calculus


# Colonnes SANS ordre naturel -> one-hot + test global du rapport de vraisemblance.
# À AJUSTER : ajoute "Lesion_num" et/ou "functional_level" ici SI elles sont nominales.
# Si functional_level est une vraie ordinale (sévérité croissante), laisse-la HORS de
# cette liste pour la traiter en continu (1 colonne + hypothèse de linéarité assumée).
NOMINAL_COLS = ["Neurol_cond"]


def load_data(data_path, cols_to_keep):
    data = pd.read_excel(data_path)

    # Sex binaire -> 0/1 (sans effet sur l'inférence, simple commodité)
    data["Sex"] = data["Sex"].replace(["M", "F"], [0, 1])

    # Neurol_cond reste en LIBELLÉS (BM/AVC/Autre) : get_dummies produira des noms lisibles.
    # recodage en 1/2/3 imposait à tort une linéarité sur un code arbitraire.

    data = data.apply(DataCleaning.lesion_level_to_num, axis=1)

    y = Calculus.calculate_MCID_2(data, 45)["MCID_classes"].rename("MCID")

    X = data[cols_to_keep].copy()

    # PAS de dropna global ici : chaque test droppera par paire (voir _pairwise).
    return X, y


def _pairwise(x, y):
    """Aligne une feature et la cible, supprime les NaN sur CE couple uniquement."""
    df = pd.concat([x.rename("x"), y.rename("MCID")], axis=1).dropna(axis=0)
    return df["x"], df["MCID"].astype(int)


def screen_continuous(x, y):
    """Logit à 1 prédicteur continu : Wald p, OR/unité, OR/écart-type, AUC apparente, MW."""
    xx, yy = _pairwise(x, y)
    xx = xx.astype(float)

    Xc = sm.add_constant(xx)
    res = sm.Logit(yy, Xc).fit(disp=0)

    beta = res.params["x"]
    or_unit = np.exp(beta)                 # OR par unité brute (par mètre, par année, ...)
    or_sd = np.exp(beta * xx.std())        # OR par écart-type -> COMPARABLE entre variables

    probs = res.predict(Xc)
    auc = roc_auc_score(yy, probs)         # IN-SAMPLE -> optimiste (cf. nom de colonne)

    # Mann-Whitney : effet brut, équivalent à l'AUC pour un prédicteur continu
    g0, g1 = xx[yy == 0], xx[yy == 1]
    mw_p = stats.mannwhitneyu(g0, g1).pvalue

    return dict(n=len(yy), type="continuous", AUC_apparent=auc,
                p_value=res.pvalues["x"], MW_p=mw_p,
                OR_per_unit=or_unit, OR_per_SD=or_sd)


def screen_nominal(x, y):
    """Variable nominale : one-hot + test du rapport de vraisemblance (k-1 ddl)."""
    xx, yy = _pairwise(x, y)
    dummies = pd.get_dummies(xx.astype("category"), drop_first=True, dtype=float)

    if dummies.shape[1] == 0:  # une seule modalité présente après dropna
        return dict(n=len(yy), type="nominal", AUC_apparent=np.nan,
                    p_value=np.nan, MW_p=np.nan,
                    OR_per_unit=np.nan, OR_per_SD=np.nan)

    const = pd.DataFrame({"const": 1.0}, index=dummies.index)
    res_null = sm.Logit(yy, const).fit(disp=0)                    # modèle constante seule
    res_full = sm.Logit(yy, sm.add_constant(dummies)).fit(disp=0)

    lr_stat = 2 * (res_full.llf - res_null.llf)
    ddl = dummies.shape[1]                                        # = k - 1
    p_lr = stats.chi2.sf(lr_stat, ddl)

    probs = res_full.predict(sm.add_constant(dummies))
    auc = roc_auc_score(yy, probs)

    # OR non réductible à un seul nombre (un OR par modalité) -> NaN dans la table de synthèse.
    # Les OR par modalité sont dans res_full.params si tu veux les détailler.
    return dict(n=len(yy), type="nominal", AUC_apparent=auc,
                p_value=p_lr, MW_p=np.nan,
                OR_per_unit=np.nan, OR_per_SD=np.nan)


def main(data_path, cols_to_keep, dataset, nominal_cols=NOMINAL_COLS,
         out_dir="results/uni_multi_variate"):
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_data(data_path, cols_to_keep)
    print(f"n total chargé = {len(y)} | "
          f"répondeurs = {int((y == 1).sum())} | non-répondeurs = {int((y == 0).sum())}")

    results = []
    for col in cols_to_keep:
        try:
            r = (screen_nominal(X[col], y) if col in nominal_cols
                 else screen_continuous(X[col], y))
        except Exception as e:  # convergence, séparation parfaite, etc.
            print(f"[WARN] {col} : {e}")
            r = dict(n=np.nan, type="error", AUC_apparent=np.nan,
                     p_value=np.nan, MW_p=np.nan,
                     OR_per_unit=np.nan, OR_per_SD=np.nan)
        r["Biomarker"] = col
        results.append(r)

    df = (pd.DataFrame(results)
          .set_index("Biomarker")
          .sort_values("AUC_apparent", ascending=False))

    print("\n" + df.to_markdown())

    df.to_pickle(os.path.join(out_dir, f"univariate_results_{dataset}.pkl"))
    df.to_excel(os.path.join(out_dir, f"univariate_results_{dataset}.xlsx"))
    return df


if __name__ == "__main__":
    data_path = ("/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/"
                 "Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/"
                 "CHUNantes collaboration/donnees/data_from_dpi/"
                 "merged_data_final.xlsx")
    # profile dataset
    # cols_to_keep = [
    #     "Neurol_cond",
    #     "Lesion_num",
    #     "Nb sessions",
    #     "Sex",
    #     "Age",
    #     "BMI",
    #     "6MWT_m_pre",
    #     "10MWT_pas_pre",
    #     "10MWT_sec_pre",
    #     "delay_injury",
    #     "delay_loko",
    #     "functional_level",
    # ]

    # merged dataset
    cols_to_keep = [
        "nb_sessions",
        # "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "Neurol_cond",
        "Sex",
        "Age",
        "Nb sessions",
        "functional_level",
        "Lesion_num",
        "BMI",
    ]

    main(data_path, cols_to_keep, dataset="merged")