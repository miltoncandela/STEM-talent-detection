import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Target variables that are not currently used, would be dropped.
x = pd.read_csv('processed/combined_df.csv')
y = x.pop('MCE_Score')
x = x.iloc[:, :-3]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=50, shuffle=False)
corr = [np.abs(np.corrcoef(x_train[feature], y_train)[0][1]) for feature in x.columns]
s_corr = pd.Series(index=x_train.columns, data=corr).sort_values(ascending=False)
print(s_corr)
# x = x.loc[:, s.index[:5]]
# x_train['Score'] = y_train
# sns.pairplot(x, hue='Score', palette=sns.diverging_palette(20, 220, n=len(np.unique(x.Score))))
# plt.scatter(x=x.iloc[:, 0], y=x.Score)
# plt.show()

# x_train.drop('Score', axis=1, inplace=True)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

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

# train_scores = [r2_score(y_train, RandomForestRegressor(random_state=50, n_estimators=20, max_depth=20, n_jobs=-1, min_samples_leaf=20).fit(x_train, y_train).predict(x_train)) for trees in range(2,100,2)]
# print('Listo train')
# test_scores = [r2_score(y_test, RandomForestRegressor(random_state=50, n_estimators=20, max_depth=20, n_jobs=-1, min_samples_leaf=20).fit(x_train, y_train).predict(x_test)) for trees in range(2,100,2)]
# print('Listo test')
#print(train_scores)
#print(test_scores)
#plt.scatter(x=range(2, 100, 2), y=train_scores, c='b', label='Train r2')
#plt.scatter(x=range(2, 100, 2), y=test_scores, c='r', label='Test r2')
#plt.xlabel('Number of trees')
#plt.ylabel('r2 Score')
#a = plt.gca()
#a.set_ylim([0, 1])
#plt.legend()
#plt.show()

reg = LinearRegression().fit(x_train, y_train)
print(r2_score(y_train, reg.predict(x_train)), r2_score(y_test, reg.predict(x_test)))
df = pd.DataFrame({'True_values': y_test, 'Predicted_values': reg.predict(x_test)}).sort_values(by='True_values', ascending=True)
plt.scatter(x=range(x_test.shape[0]), y=df['True_values'], c='b', label='True')
plt.scatter(x=range(x_test.shape[0]), y=df['Predicted_values'], c='r', label='Predicted')
plt.legend()
plt.show()
