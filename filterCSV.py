import numpy as np
import pandas as pd

# Sklearn
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE

np.seterr(divide='ignore', invalid='ignore')


def filter_csv(metric, n_features=100):
    """
    This function takes the combined feature CSV, created using "combine_dfs" and "feature_generation" in "createCSV.py"
    and a specific continuous target variable. The function then removes correlated features and starts a feature
    selection process using RandomForestRegressor and computing the GINI coefficient, the best columns are used as a
    subset on the whole DataFrame, creating a smaller CSV within the "processed" folder.


    :param string metric: Either MCE_Score or PSI_Score, the continuous target variable that would be used.
    :param integer n_features: Top n_features features that would be considered for the filtered CSV.
    """

    # The whole DataFrame is read, invalid IDs that have NANs are removed, and information is conserved.
    x = pd.read_csv('processed/combined_df.csv', dtype={'MCE_Category': 'category', 'PSI_Category': 'category',
                                                        'ID': 'string', 'Take': int})
    x = x.drop(x[x.ID.isin(NOT_VALID_IDS)].index, axis=0)
    df_info = x[['ID', 'Take','PSI_Score', 'MCE_Score', 'MCE_Category', 'PSI_Category']]
    print(Counter(list(x.MCE_Category)))
    print(Counter(list(x.PSI_Category)))

    # The desired target variable is popped form the main DataFrame and information variables are removed from it.
    y = x.pop(metric)
    x = x.drop(['ID', 'Take'], axis=1).iloc[:, :-3].astype(np.float32)
    # x, y = SMOTE(random_state=50).fit_resample(x, y)

    # A correlation matrix is done, to remove highly correlated features, which commonly appear when feature generation.
    corr_matrix = x.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    x = x.drop([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)], axis=1)

    # The following for loop iterates through a set of MAX_ITER seeds, and fits a RandomForestRegressor multiple times
    # on the same DataFrame. The reason behind this is due to the fact that the RF model is based on boosting and it
    # has a randomness factor related, which is controlled via the "random_state" parameter. So, by computing feature
    # importance when using multiple seeds, we increase its generalization power.
    df_results = pd.DataFrame(columns=x.columns)
    for seed in range(1, MAX_ITER):
        print(round(seed/MAX_ITER * 100, 2), '% ...')
        model = RandomForestRegressor(random_state=seed).fit(x, y)
        df_results = df_results.append(dict(zip(list(x.columns), model.feature_importances_)), ignore_index=True)
    print(df_results.head())
    s = df_results.sum().sort_values(ascending=False)
    print(s)

    # s = pd.Series(index=x.columns, data=[np.abs(np.corrcoef(x[feature], y)[0][1]) for feature in x.columns])\
    #     .sort_values(ascending=False)
    # print(s)

    x = pd.concat([x.loc[:, list(s.index[:n_features])], df_info], axis=1)
    x.to_csv('processed/filtered_{}.csv'.format(metric), index=False)


NOT_VALID_IDS = ['EJ01', 'ST02']
MAX_ITER = 25
filter_csv('MCE_Score')
# filter_csv('PSI_Score')
