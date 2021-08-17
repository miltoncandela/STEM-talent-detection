import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import dump

# Sklearn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from imblearn.under_sampling import RandomUnderSampler

# Target variables that are not currently used, would be dropped.
MCE_categories = {2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
PSI_categories = {0: 'Negativo', 1: 'Positivo'}
N_FEATURES = 25

X = pd.read_csv('processed/combined_df.csv', dtype={'MCE_Category': 'category', 'PSI_Category': 'category',
                                                    'ID': 'string', 'Take': int}).drop(['ID', 'Take'], axis=1)

X = X.dropna(axis='rows', how='any').drop(X[X.MCE_Category == 'Insuficiente'].index, axis=0)
print(Counter(list(X.MCE_Category)))
X, _ = RandomUnderSampler(random_state=50).fit_resample(X, X.MCE_Category)
print(Counter(list(X.MCE_Category)))

y = X.pop('MCE_Score')
# y = pd.Categorical(y, categories=list(MCE_categories.values()), ordered=True)

X = X.iloc[:, :-3].replace([np.inf, -np.inf], np.nan).dropna(axis='columns', how='any').astype(np.float32)
print(X.shape)

corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X = X.drop(to_drop, axis=1)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=50, shuffle=True)

model = RandomForestRegressor(random_state=50).fit(X, y)
s = pd.Series(index=X.columns, data=model.feature_importances_).sort_values(ascending=False)
print(s)
X_train, X_test = X_train.loc[:, s.index[:N_FEATURES]], X_test.loc[:, s.index[:N_FEATURES]]
file = open('processed/MCE_Score_Cols.txt', 'w')
file.write('\n'.join(list(s.index)))
file.close()
exit()
'''
MAX_FEATURES = 200
df_results = pd.DataFrame(columns=['LIR_Train', 'LIR_Test', 'LAS_Train', 'LAS_Test',
                                   'KNN_Train', 'KNN_Test', 'RF_Train', 'RF_Test',
                                   'GBM_Train', 'GBM_Test'])
for N_FEATURES in range(2, MAX_FEATURES):
    print(round(N_FEATURES/MAX_FEATURES * 100, 2), '% ...')
    X_train_filt, X_test_filt = X_train.loc[:, s.index[:N_FEATURES]], X_test.loc[:, s.index[:N_FEATURES]]

    results = []

    model = LinearRegression().fit(X_train_filt, y_train)
    results.append(r2_score(y_train, model.predict(X_train_filt)))
    results.append(r2_score(y_test, model.predict(X_test_filt)))

    model = Lasso().fit(X_train_filt, y_train)
    results.append(r2_score(y_train, model.predict(X_train_filt)))
    results.append(r2_score(y_test, model.predict(X_test_filt)))

    model = KNeighborsRegressor().fit(X_train_filt, y_train)
    results.append(r2_score(y_train, model.predict(X_train_filt)))
    results.append(r2_score(y_test, model.predict(X_test_filt)))

    model = RandomForestRegressor(random_state=50).fit(X_train_filt, y_train)
    results.append(r2_score(y_train, model.predict(X_train_filt)))
    results.append(r2_score(y_test, model.predict(X_test_filt)))

    model = GradientBoostingRegressor(random_state=50).fit(X_train_filt, y_train)
    results.append(r2_score(y_train, model.predict(X_train_filt)))
    results.append(r2_score(y_test, model.predict(X_test_filt)))

    df_results = df_results.append(dict(zip(list(df_results.columns), results)), ignore_index=True)

print(df_results)
df_results.index = range(2, MAX_FEATURES)
df_results.to_csv(path_or_buf='processed/MCE_Score_Results.csv')
exit()
'''


# x_train.drop('Score', axis=1, inplace=True)
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV

# parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': np.array(range(1, 100,1))/10}
parameters = {'n_estimators': range(25, 100, 25), 'min_samples_split': range(2, 10, 2),
              'min_samples_leaf': range(1, 5), 'max_depth': range(500, 2000, 500),
              'max_features': ['auto', 'sqrt', 'log2']}
model = GridSearchCV(RandomForestRegressor(random_state=50), parameters, cv=5, verbose=2).fit(X_train, y_train)
print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
print(r2_score(model.predict(X_train), y_train), r2_score(model.predict(X_test), y_test))
dump(model, open('processed/model.pkl', 'wb'))
file = open('processed/MCE_Score_Cols.txt', 'w')
file.write('\n'.join(list(s.index)))
file.close()
exit()

k = 5
kf = KFold(n_splits=k)

pd.options.display.float_format = '{:.6f}'.format
# pd.set_option('display.float_format', lambda x: '%.6f' % x)
df_results = pd.DataFrame(columns=['LR_Train', 'LR_Test', 'RF_Train', 'RF_Test', 'GBR_Train', 'GBR_Test'])

for train_index, test_index in kf.split(x):
    x_train_fold, x_test_fold = x.iloc[train_index, :], x.iloc[test_index, :]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    lr = LinearRegression().fit(x_train_fold, y_train_fold)
    rf = RandomForestRegressor(random_state=50, n_estimators=20, max_depth=20).fit(x_train_fold, y_train_fold)
    gbr = GradientBoostingRegressor().fit(x_train_fold, y_train_fold)

    metrica = [r2_score, mean_squared_error][1]

    results = dict(zip(list(df_results.columns), [metrica(y_train_fold, lr.predict(x_train_fold)),
                                                    metrica(y_test_fold, lr.predict(x_test_fold)),
                                                    metrica(y_train_fold, rf.predict(x_train_fold)),
                                                    metrica(y_test_fold, rf.predict(x_test_fold)),
                                                    metrica(y_train_fold, gbr.predict(x_train_fold)),
                                                    metrica(y_test_fold, gbr.predict(x_test_fold))]))
    df_results = df_results.append(results, ignore_index=True)

print(df_results)
exit()
print(r2_score(y_train, reg.predict(x_train)), r2_score(y_test, reg.predict(x_test)))
print(r2_score(y_train, reg.predict(x_train)), r2_score(y_test, reg.predict(x_test)))
