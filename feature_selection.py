import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from collections import Counter


def feature_selection(n):
    """
    Reads a CSV file on the Processed folder, depending on the name provided, then it creates a text file from all
    the features from most important to least important. It is separated with a escape sequence to further create a
    list of the best columns and subset the whole dataset depending on the number of columns required.

    :param string name: Name of the CSV from which the data is being extracted (Paper, Gaby).
    """

    x = pd.read_csv('processed/combined_df.csv').drop(['MCE_Score', 'PSI_Score'], axis=1)
    y = x.loc[:, ['MCE_Category', 'PSI_Category']]
    x.drop(['MCE_Category', 'PSI_Category'], axis=1, inplace=True)

    top_features = []
    for column in y.columns:
        y_curr = y.loc[:, column]
        methods_dict = {'MI': mutual_info_classif(x, y_curr),
                        'RF': RandomForestClassifier(random_state=50, n_estimators=10).fit(x, y_curr).feature_importances_,
                        'FS': f_classif(x, y_curr)[0],
                        'LOG': np.mean(LogisticRegression(max_iter=50000000).fit(x, y_curr).coef_, axis=0),
                        'SVM': np.mean(LinearSVC(max_iter=50000000).fit(x, y_curr).coef_, axis=0),
                        'CHI': chi2(x, y_curr)[0],
                        'EXT': ExtraTreesClassifier(random_state=50, n_estimators=10).fit(x, y_curr).feature_importances_}

        for f_method in methods_dict:
            s = pd.Series(data=methods_dict[f_method], index=x.columns).sort_values(ascending=False)
            for element in list(s.index[:n]):
                top_features.append(element)
    print(top_features)
    s_freq = pd.Series(Counter(top_features)).sort_values(ascending=False)
    print(s_freq)


methods = ['MI', 'RF', 'FS', 'LOG', 'SVM', 'CHI', 'EXT']

# MI : Mutual Information
# RF : Random Forest
# FS : ANOVA F-value
# MI : Mutual information
# LOG : Logistic Regression
# EXT : Extremely Randomized Tree

feature_selection(15)
'''
for signal in ['Paper', 'Gaby']:
    print('Empezando con {}...'.format(signal))
    feature_selection(name=signal)
    print('Listo para {}'.format(signal))
'''