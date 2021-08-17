from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from collections import Counter
import matplotlib.pyplot as plt

skf = StratifiedKFold(n_splits=5)

MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
X = pd.read_csv('processed/filtered_MCE_Score.csv', dtype={'MCE_Category': 'category', 'PSI_Category': 'category',
                                                    'ID': 'string', 'Take': int}).drop(['ID', 'Take'], axis=1)
X = X.drop(X[X.MCE_Category == 'Insuficiente'].index, axis=0)
print(Counter(list(X.MCE_Category)))
print(Counter(list(X.PSI_Category)))

y = pd.DataFrame({'MCE_Score': X.pop('MCE_Score'), 'MCE_Category': X.pop('MCE_Category')})

N_FEATURES = 8
X = X.iloc[:, :-2].iloc[:, :N_FEATURES]
print()
print(y.head())
print(y.shape)

results_cont, results_cat = [], []

for train_index, test_index in skf.split(X, y.MCE_Category):
    X_train_fold, X_test_fold = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train_fold, y_test_fold = y.iloc[train_index, :].MCE_Score, y.iloc[test_index, :].MCE_Score

    model = RandomForestRegressor(random_state=50).fit(X_train_fold, y_train_fold)
    results_cont.append(np.abs(r2_score(y_test_fold, model.predict(X_test_fold))))

    trues = pd.cut(y_test_fold, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values())
    preds = pd.cut(model.predict(X_test_fold), [0, 1, 2, 3, 4, 5], labels=MCE_categories.values())
    results_cat.append(accuracy_score(trues, preds))

print(results_cont)
print(np.mean(results_cont))
print(results_cat)
print(np.mean(results_cat))

exit()

plt.scatter(x=range(len(l_results_cont)), y=l_results_cont, c='b', label='R2')
plt.scatter(x=range(len(l_results_cat)), y=l_results_cat, c='r', label='Accuracy')
plt.legend()
plt.show()