from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
from amelio_medullo import DataCleaning, Calculus

def load_data(data_path, cols_to_keep):
    data = pd.read_excel(data_path)
    data["Neurol_cond"] = data["Neurol_cond"].replace(["BM", "AVC", "Autre"], [1, 2, 3])
    data["Sex"] = data["Sex"].replace(["M", "F"], [1, 2])
    data.apply(DataCleaning.lesion_level_to_num, axis=1)
    X = data[cols_to_keep]
    y = Calculus.calculate_MCID_2(data, 30)
    X['MCID'] = y['MCID_classes']
    # Drops rows where the target or the specific feature is missing
    clean_X = X.dropna(axis=0)

    return clean_X.drop(columns=["MCID"], axis=1), clean_X[["MCID"]]

def simple_stats(data, y, feature):
    group0 = data[y["MCID"] == 0][feature]
    group1 = data[y["MCID"] == 1][feature]
    return stats.mannwhitneyu(group0, group1)


def main(data_path, cols_to_keep):
    
    results = []
    X, y = load_data(data_path, cols_to_keep)
    
    for col in cols_to_keep:
        # 1. Prepare data
        X_sm = sm.add_constant(X[col])
        
        # 2. Fit Logistic Regression to get P-Value and Odds Ratio
        try:
            model = sm.Logit(y, X_sm).fit(disp=0)
            p_val = model.pvalues[col]
            odds_ratio = np.exp(model.params[col])
            probs = model.predict(X_sm)
            auc = roc_auc_score(y, probs)
        except:
            # Handles cases where the model fails to converge
            p_val, odds_ratio, auc = np.nan, np.nan, np.nan
        
        _, mw_p = simple_stats(data=X, y=y, feature=col)
        
        results.append({
            'Biomarker': col,
            'AUC': auc,
            'Logit_P_Value': p_val,
            'MW_U_P_Value': mw_p,
            'Odds_Ratio': odds_ratio
        })
        print('\n'+'* '*10)
        print(f'Biomarker: {col}\nAUC: {auc}\nLogit_P_Value: {p_val}\nMW_U_P_Value: {mw_p}\nOdds_Ratio: {odds_ratio}')
    
    univariate_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    print(univariate_df.to_markdown())
    return univariate_df

if __name__ == "__main__":
    data_path = 

    # SELECTED FEATURES
    cols_to_keep = [
        "Neurol_cond",
        "Lesion_num",
        "Nb sessions",
        "Sex",
        "Age",
        "BMI",
        "6MWT_m_pre",
        "10MWT_pas_pre",
        "10MWT_sec_pre",
        "delay_injury",
        "delay_loko",
        "functional_level",
    ]

    main(data_path, cols_to_keep)