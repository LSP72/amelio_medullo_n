import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Code based on a paper by Kuo et al. (2021)


# Function to plot the ROC-AUC curve for each model
def plot_cv_roc_curve(model, X, y, cv, model_name, output_path, show=True):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(18, 16))

    # We must manually run the CV to get the curves for each fold
    for i, (train, test) in enumerate(cv.split(X, y)):
        x_train, x_test = X.iloc[train], X.iloc[test]
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model.fit(x_train_scaled, y.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            model, x_test_scaled, y.iloc[test], name=f"ROC fold {i}", alpha=0.3, lw=1, ax=ax
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Plot the Luck line
    ax.plot([0, 1], [0, 1], "r--", label="Chance", alpha=0.8)

    # Calculate and plot Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})", lw=2, alpha=0.8
    )

    # Shade the confidence interval (Stability)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label="$\pm$ 1 std. dev.")

    ax.set(
        xlabel="False Positive Rate (1 - Specificity)",
        ylabel="True Positive Rate (Sensitivity)",
        title=f"ROC Curve: {model_name}",
    )
    ax.legend(loc="lower right")
    plt.savefig(f"{output_path}/ROC_Curve_{model_name.replace(' ', '_')}.png", dpi=300)
    if show == True:
        plt.show()


# %% DATA LOADING & PREPROCESSING
data = pd.read_excel("/Volumes/SP UFD U2/PhD/Stage Nantes/LOKOMAT/reports_final_table.xlsx")
output_path = "/Users/mathildetardif/Documents/Python/Biomarkers/amelio_medullo_n/results/Intervention_results/Model_Comparison_Results"
data.dropna(subset=["MCID_6MWT"], inplace=True)  # Remove rows with missing target variable
y = data["MCID_6MWT"]
X = data.drop(columns=["MCID_6MWT"])

# %%MACHINE LEARNING MODELS INITIALISATION
# 3 classifiers have been compared
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": GradientBoostingClassifier(n_estimators=100, random_state=42),
}
cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)  # stratifying to assess the stability

results = {}  # Dictionary to store the results for comparison

# %% MODELS' TRAINING & EVALUATION
# Using area under the curve (AUC) to assess the models

plt.figure(figsize=(8, 6))

for name, model in models.items():
    print("\n" + "* " * 20)
    print(f"Running for {name}")

    # Build the Pipeline
    stability_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)), (name, model)]
    )  # Pipeline ensures SMOTE only applied to the training folds in EACH split

    print("Running Repeated Cross-Validation...")
    scoring_metrics = ["roc_auc", "accuracy"]

    # Use cross_validate instead of cross_val_score
    cv_results = cross_validate(stability_pipeline, X, y, scoring=scoring_metrics, cv=cv_strategy, n_jobs=-1)

    results[name] = cv_results["test_roc_auc"]

    print(f"\n--- Model Stability of {name} ---")
    print(f"Mean Accuracy:      {cv_results['test_accuracy'].mean():.4f}")
    print(f"Stability (Std Dev): {cv_results['test_accuracy'].std():.4f}")
    print(f"Mean AUC:           {cv_results['test_roc_auc'].mean():.4f}")
    print(f"Stability (Std Dev): {cv_results['test_roc_auc'].std():.4f}")
    print(f"Median AUC:         {np.median(cv_results['test_roc_auc']):.4f}")
    print(f"Minimum AUC observed: {cv_results['test_roc_auc'].min():.4f}")
    print(f"Maximum AUC observed: {cv_results['test_roc_auc'].max():.4f}")

    plot_cv_roc_curve(stability_pipeline, X, y, cv_strategy, name, output_path)

    # fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

# %% VISUALISATION OF THE MODELS' VARIABILITY
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=results_df, palette="pastel", showfliers=False  # Outliers hidden, so won't overlap with the stripplot points
)

sns.stripplot(data=results_df, color="black", alpha=0.6, jitter=True, size=6)

plt.title(f"Assessment of Model Stability Across 25 Random Data Splits", fontsize=14, pad=15)
plt.ylabel("ROC AUC Score", fontsize=12)
plt.xlabel("Machine Learning Algorithm", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0.4, 1.05)

plt.tight_layout()
plt.show()


# %% FEATURE IMPORTANCE ANALYSIS (if RF was good)

# rf_model = models['Random Forest']
# feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# print("Top Predictors of Ambulatory Progress:")
# print(feature_importances.sort_values(ascending=False).head(5))
