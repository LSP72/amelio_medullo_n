import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from amelio_medullo import Calculus


def lasso_feature_selection(X, y, classif, test_size=0.2, random_state=42, cv=5):
    """
    Uses LassoCV to select important biomarkers/features.

    Returns:
        selected_features
        coefficients dataframe
        fitted model
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if classif:
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',  # obligatoire pour L1 en binaire
            C=0.1,
            max_iter=10000
        )
        model.fit(X_train, y_train)
        coefs = pd.Series(model.coef_[0], index=X_train.columns.to_list())
        biomarqueurs = coefs[coefs != 0].sort_values(key=abs, ascending=False)
    
        print(coefs)
        print("Number of selected features:", len(biomarqueurs))

    else:
        # Elastic net
        model = Pipeline(
            [("scaler", StandardScaler()), ("lasso", ElasticNetCV(cv=cv, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], random_state=random_state, max_iter=10000, alphas=np.logspace(-6, 0, 200)))]
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        lasso = model.named_steps["lasso"]

        coef_df = pd.DataFrame({"Feature": X.columns.to_list(), "Coefficient": lasso.coef_})

        coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

        selected_features = coef_df[coef_df["Coefficient"] != 0]["Feature"].tolist()

        print("Best alpha:", lasso.alpha_)
        print("Test R²:", r2)
        print("Test RMSE:", rmse)
        print("Number of selected features:", len(selected_features))

    # return selected_features, coef_df, model


def main(data_path, cols_to_keep, classif=True):
    data = pd.read_excel(data_path)
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    X = data[cols_to_keep]

    if classif:
        y = data["MCID_classes"]
    else:
        y = data["6MWT_m_post"]
    
    lasso_feature_selection(X, y, classif)
    
    


if __name__ == "__main__":
    data_path = 
    ## SELECTED FEATURES for loko
    cols_to_keep = [
        "nb_sessions",
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "6MWT_m_pre",
        "functional_level",
        "cadence",
    ]
    ## ALL FEATURES for loko
    # cols_to_keep = ["nb_sessions",	"duration",	"Distance_m",	"Distance_pas",	"Durée_min",	"Vitesse_kmh_MIN",	"Vitesse_kmh_MAX",	"Vitesse_kmh_MOY",	"BWS_%_MIN",	"BWS_%_MAX",
    #                      "BWS_%_MOY",	"BWS_kg_MIN",	"BWS_kg_MAX",	"BWS_kg_MOY",	"Guidage_G_%_MIN",	"Guidage_G_%_MAX",	"Guidage_G_%_MOY",	"Guidage_D_%_MIN",	"Guidage_D_%_MAX",
    #                      "Guidage_D_%_MOY",	"sessions_per_week",	"6MWT_m_pre", "functional_level", "cadence"]
    # features for merged data
    cols_to_keep = [
        "nb_sessions",
        "duration",
        "Durée_min",
        "Vitesse_kmh_MOY",
        "BWS_%_MOY",
        "step_length",
        "Guidage_%_MOY",
        "sessions_per_week",
        "Neurol_cond",
        "Sex",
        # "Age", # 1 missing
        "Nb sessions",
        "functional_level",
        # "Lesion_num", # 5 missing
        # "BMI", # 7 missing
    ]
    random_state_list = [42]
    # random_state_list = np.arange(1, 101)
    output_path = "results/loko_results"

    main(data_path, cols_to_keep, classif=True)
