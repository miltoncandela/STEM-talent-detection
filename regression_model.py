import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from collections import Counter
from matplotlib.offsetbox import AnchoredText

# Target variables that are not currently used, would be dropped.
MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
x = pd.read_csv('processed/filtered_MCE_Score.csv').drop(['ID', 'Take'], axis=1)
y = x.pop('MCE_Score')
N_FEATURES = 17
x = x.iloc[:, :-3].iloc[:, :N_FEATURES]
print(x.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=50)

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x_train, y_train)
print(r2_score(y_test, model.predict(x_test)))

model = GradientBoostingRegressor(random_state=50).fit(x_train, y_train)
print(r2_score(y_test, model.predict(x_test)))

model = RandomForestRegressor(random_state=50).fit(x_train, y_train)
print(r2_score(y_test, model.predict(x_test)))


# print(accuracy_score(pd.cut(y_test, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values()),
#                      pd.cut(model.predict(x_test), [0, 1, 2, 3, 4, 5], labels=MCE_categories.values())))
# trues = pd.cut(y_test, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values())
# preds = pd.cut(model.predict(x_test), [0, 1, 2, 3, 4, 5], labels=MCE_categories.values())

df = pd.DataFrame({'True_values': y_test, 'Predicted_values': model.predict(x_test)}).sort_values(by='True_values', ascending=True)
fig = plt.figure()
plt.scatter(x=range(x_test.shape[0]), y=df['True_values'], c='b', label='Reference')
plt.scatter(x=range(x_test.shape[0]), y=df['Predicted_values'], c='r', label='Predicted')
a = plt.gca()
a.set_title('Reference and predicted STEM interest using RF with 17 features')
a.set_xlabel("Sample's index using testing dataset from 80:20 split")
a.set_ylabel('Delta of STEM interest, given pre and pos evaluation')
plt.legend()
at = AnchoredText("Coefficient of determination $(R^2)$: {}".format(round(r2_score(y_test, model.predict(x_test)), 2)),
                  loc='lower right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
a.add_artist(at)
plt.show()
# fig.savefig('figures/results/PSI_prediction.png', bbox_inches='tight')

# prop=dict(size=15), frameon=True,

trues = pd.Categorical(['Positive' if score > 0 else 'Negative' for score in y_test], categories=['Positive', 'Negative'])
preds = pd.Categorical(['Positive' if score > 0 else 'Negative' for score in model.predict(x_test)], categories=['Positive', 'Negative'])
print(confusion_matrix(trues, preds))
print(Counter(trues), Counter(preds))
# sns.heatmap(confusion_matrix(trues, preds, labels=['Positive', 'Negative']), annot=True)
# plt.matshow(confusion_matrix(trues, preds, labels=['Positive', 'Negative'], normalize='true'))
# plt.show()
# print(confusion_matrix(pd.cut(y_test, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values()),
#                        pd.cut(model.predict(x_test), [0, 1, 2, 3, 4, 5], labels=MCE_categories.values())))
exit()

# x_train.drop('Score', axis=1, inplace=True)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression

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
