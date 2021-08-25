# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code creates a filtered CSV, according to the importance of the features to the target variable using
# RandomForestRegressor, even though no sampling technique was employed to solve the class imbalance, this was due to
# the fact that a regression approach was taken rather than a classification. The dataset is saved as a CSV file inside
# the "processed" folder, with the top 100 features and their respective information variables.

# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code creates a filtered CSV, according to the importance of the features to the target variable using
# RandomForestRegressor, even though no sampling technique was employed to solve the class imbalance, this was due to
# the fact that a regression approach was taken rather than a classification. The dataset is saved as a CSV file inside
# the "processed" folder, with the top 100 features and their respective information variables.

# Data Treatment
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
    x = pd.read_csv('processed/combined_df_2.csv', dtype={'MCE_Category': 'category', 'PSI_Category': 'category',
                                                          'ID': 'string', 'Take': int})
    x = x.drop(x[x.ID.isin(NOT_VALID_IDS)].index, axis=0)
    df_info = x[['ID', 'Take', 'PSI_Score', 'MCE_Score', 'MCE_Category', 'PSI_Category']]
    print(Counter(list(x.MCE_Category)))
    print(Counter(list(x.PSI_Category)))

    # The desired target variable is popped form the main DataFrame and information variables are removed from it.
    y = x.pop(metric)
    x = x.drop(['ID', 'Take'], axis=1).iloc[:, :-3].astype(np.float32)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=50)

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
    s = df_results.sum().sort_values(ascending=False)

    # The best n_features are being used to create the filtered DataFrame, which also includes "df_info" in order to
    # use it on next experiments and tests, instead of loading the whole dataset of combined features.
    x = pd.concat([x.loc[:, list(s.index[:n_features])], df_info], axis=1)
    x.to_csv('processed/filtered_{}_3.csv'.format(metric), index=False)


# The following constants are being created: NOT_VALID_IDS (which are students who dropped from the experiment) and
# MAX_ITER, that would be used as the number of iterations made for our feature selection process.
NOT_VALID_IDS = ['EJ01', 'ST02']
MAX_ITER = 5
for score in ['MCE_Score', 'PSI_Score']:
    filter_csv(score)
