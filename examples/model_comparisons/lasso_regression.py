import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from amelio_medullo import Calculus


def lasso_feature_selection(X, y, test_size=0.2, random_state=42, cv=5):
    """
    Uses LassoCV to select important biomarkers/features.

    Returns:
        selected_features
        coefficients dataframe
        fitted model
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = Pipeline(
        [("scaler", StandardScaler()), ("lasso", LassoCV(cv=cv, random_state=random_state, max_iter=10000))]
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

    return selected_features, coef_df, model


def main(data, cols_to_keep):
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    X = data[cols_to_keep]
    y = Calculus.calculate_MCID_2(data, 45)

    lasso_feature_selection(X, y)


if __name__ == "__main__":
    data_path = 
    data = pd.read_excel(data_path)
    cols_to_keep = [
        "Neurol_cond",
        "Lesion_num",
        "Sex",
        "Age",
        "Height",
        "Weight",
        "6MWT_m_pre",
        "10MWT_pas_pre",
        "10MWT_sec_pre",
        "6MWT_m_post",
        "delay_injury",
        "delay_loko",
        "functional_level",
        "Artic_hip_flex",
        "Artic_hip_ext",
        "Artic_hip_add",
        "Artic_hip_abd",
        "Artic_hip_rot_ext",
        "Artic_hip_rot_int",
        "Knee_flex",
        "Knee_ext",
        "Ank_flex_90",
        "Ank_flex_180",
        "Ank_ext",
        "H_Flex_ass",
        "H_Ext_PP",
        "H_abd",
        "H_add",
        "H_rot_int",
        "K_Flex",
        "K_Ext",
        "A_Dorsiflex_GT",
        "A_Plantarflex",
    ]

    main(data, cols_to_keep)
