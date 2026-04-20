from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import shap
import matplotlib.pyplot as plt
import time
from amelio_medullo import Calculus
import pandas as pd
import numpy as np

print(time.time())

data = pd.read_excel("/Users/mathildetardif/Documents/Documents/PhD/Nantes/data_drafts/prelim_table_2.xlsx")
MCID = Calculus.calculate_MCID(data["6MWT_m_pre"], data["6MWT_m_post"], threshold=45)
data.drop(columns=["IPP", "6MWT_m_post", "Height", "Weight", "Nb sessions", "ASIA_mot_D_pre", "ASIA_mot_G_pre", "ASIA_mot_pre", "ASIA_LL_D", "ASIA_LL_G", "ASIA_LL", "10MWT_pas_pre"], inplace=True)
X, y = data, MCID
data["Trouble neuro"] = data["Trouble neuro"].replace(["AVC", "BM", "Autre"], [1, 2, 3])
features_names = X.columns.to_list()

imp_mean = IterativeImputer(missing_values=np.nan, n_nearest_features=5, random_state=42, imputation_order="ascending")
imp_mean.fit(X)
X = imp_mean.transform(X)

def shap_analysis(model, x_train, x_test, features_names, model_name, strat):
    explainer = shap.KernelExplainer(model.predict, x_train)
    shap_values = explainer.shap_values(x_test)

    plt.figure(figsize=(8, 10))
    shap.summary_plot(
        shap_values,
        x_test,
        feature_names=features_names,  # model.feature_keys
        max_display=len(features_names),
        plot_size=(8, 10),
        show=False,  # Prevent SHAP from auto-displaying
    )
    # save_path = f"/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/model_tests/shaps/{model_name}_{strat}_shap_plot.png"
    # plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


stratify = input(
    "Use of Stratified K-Fold? (True/False): "
)  # Set to True for stratified splitting, False for random splitting without stratification
loo = input(
    "Use Leave-One-Out Cross-Validation? (True/False): "
)  # Set to True for Leave-One-Out CV, False for Stratified K-Fold CV
if loo == "True":
    Kstrat = LeaveOneOut()
else:
    Kstrat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


if stratify == "True":
    print("*" * 20)
    print("Using stratified splitting based on the target variable.")

    print(f"\n\n===== Processing =====\n")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    ## Logistic Regression
    log_reg = LogisticRegression(max_iter=2000, random_state=42)
    log_reg.fit(X, y)
    log_reg_score = cross_val_score(log_reg, X, y, cv=Kstrat)
    print(f"Logistic Regression Accuracy: {log_reg_score.mean():.4f} (+/- {log_reg_score.std() * 2:.4f})")
    # shap_analysis(log_reg, x_train, x_test, features_names)

    ## K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn_score = cross_val_score(knn, X, y, cv=Kstrat)
    print(f"K-Nearest Neighbors Accuracy: {knn_score.mean():.4f} (+/- {knn_score.std() * 2:.4f})")
    # shap_analysis(knn, x_train, x_test, features_names)

    ## Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt_score = cross_val_score(dt, X, y, cv=Kstrat)
    print(f"Decision Tree Accuracy: {dt_score.mean():.4f} (+/- {dt_score.std() * 2:.4f})")
    # shap_analysis(dt, x_train, x_test, features_names)

    ## Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_score = cross_val_score(rf, X, y, cv=Kstrat)
    print(f"Random Forest Accuracy: {rf_score.mean():.4f} (+/- {rf_score.std() * 2:.4f})")
    # shap_analysis(rf, x_train, x_test, features_names)

    ## HistGradientBoosting
    hgb = HistGradientBoostingClassifier(random_state=42)
    hgb_score = cross_val_score(hgb, X, y, cv=Kstrat)
    print(f"HistGradientBoosting Accuracy: {hgb_score.mean():.4f} (+/- {hgb_score.std() * 2:.4f})")
    # shap_analysis(hgb, x_train, x_test, features_names)

    ## Support Vector Machine
    svc = SVC(random_state=42)
    svc_score = cross_val_score(svc, X, y, cv=Kstrat)
    print(f"Support Vector Machine Accuracy: {svc_score.mean():.4f} (+/- {svc_score.std() * 2:.4f})")
    # shap_analysis(svc, x_train, x_test, features_names)

    ## Linear Support Vector Machine
    linear_svc = LinearSVC(random_state=42)
    linear_svc_score = cross_val_score(linear_svc, X, y, cv=Kstrat)
    print(
        f"Linear Support Vector Machine Accuracy: {linear_svc_score.mean():.4f} (+/- {linear_svc_score.std() * 2:.4f})"
    )
    # shap_analysis(linear_svc, x_train, x_test, features_names)

else:
    print(f"\n\n===== Processing =====\n")
    strats = [None, X[:,0]]  # No stratification and stratification by the target variable

    for strat in strats:
        if strat is None:
            print("\n--- No Stratification ---\n")
        else:
            print("\n--- Stratification by Target Variable ---\n")

        x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=strat, test_size=0.2, random_state=42)

        ## Logistic Regression
        log_reg = LogisticRegression(max_iter=2000, random_state=42)
        log_reg.fit(x_train, y_train)
        log_reg_score = log_reg.score(x_test, y_test)
        print(f"Logistic Regression Accuracy: {log_reg_score:.4f}")
        shap_analysis(log_reg, x_train, x_test, features_names, "log_reg", "strat" if strat is not None else "no_strat")

        ## K-Nearest Neighbors
        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        knn_score = knn.score(x_test, y_test)
        print(f"K-Nearest Neighbors Accuracy: {knn_score:.4f}")
        shap_analysis(knn, x_train, x_test, features_names, "knn", "strat" if strat is not None else "no_strat")

        ## Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(x_train, y_train)
        dt_score = dt.score(x_test, y_test)
        print(f"Decision Tree Accuracy: {dt_score:.4f}")
        shap_analysis(dt, x_train, x_test, features_names, "dt", "strat" if strat is not None else "no_strat")

        ## Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(x_train, y_train)
        rf_score = rf.score(x_test, y_test)
        print(f"Random Forest Accuracy: {rf_score:.4f}")
        shap_analysis(rf, x_train, x_test, features_names, "rf", "strat" if strat is not None else "no_strat")

        ## HistGradientBoosting
        hgb = HistGradientBoostingClassifier(random_state=42)
        hgb.fit(x_train, y_train)
        hgb_score = hgb.score(x_test, y_test)
        print(f"HistGradientBoosting Accuracy: {hgb_score:.4f}")
        shap_analysis(hgb, x_train, x_test, features_names, "hgb", "strat" if strat is not None else "no_strat")

        ## Support Vector Machine
        svc = SVC(random_state=42)
        svc.fit(x_train, y_train)
        svc_score = svc.score(x_test, y_test)
        print(f"Support Vector Machine Accuracy: {svc_score:.4f}")
        shap_analysis(svc, x_train, x_test, features_names, "svc", "strat" if strat is not None else "no_strat")

        # ## Support Vector Machine with my OOP
        # svc_model = SVCModel()
        # svc_model.add_data(X, y, test_size=0.2)
        # svc_model.model.fit(svc_model.X_train, svc_model.y_train)
        # svc_score = svc_model.model.score(svc_model.X_test, svc_model.y_test)
        # print(f"SVCModel Accuracy: {svc_score:.4f}")

        # ## Support Vector Machine with my OOP + BO
        # svc_model_bo = SVCModel()
        # svc_model_bo.add_data(X, y, test_size=0.2)
        # svc_model_bo.train_and_tune("bayesian_optim")
        # svc_score = svc_model_bo.model.score(svc_model_bo.X_test, svc_model_bo.y_test)
        # print(f"SVCModel Accuracy: {svc_score:.4f}")

        ## Linear Support Vector Machine
        linear_svc = LinearSVC(random_state=42)
        linear_svc.fit(x_train, y_train)
        linear_svc_score = linear_svc.score(x_test, y_test)
        print(f"Linear Support Vector Machine Accuracy: {linear_svc_score:.4f}")
        shap_analysis(linear_svc, x_train, x_test, features_names, "linear_svc", "strat" if strat is not None else "no_strat")
