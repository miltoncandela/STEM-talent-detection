import pandas as pd
import numpy as np
from pickle import load
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
X = pd.read_csv('processed/filtered_MCE_Score.csv', dtype={'MCE_Category': 'category', 'PSI_Category': 'category',
                                                    'ID': 'string', 'Take': int})
print(Counter(list(X.MCE_Category)))
print(Counter(list(X.PSI_Category)))
# X, _ = RandomUnderSampler(random_state=50).fit_resample(X, X.MCE_Category)
# print(Counter(list(X.MCE_Category)))

y = pd.DataFrame({'MCE_Score': X.pop('MCE_Score'), 'ID': X.ID, 'Take': X.Take})
print(y.head())
print(y.shape)

N_FEATURES = 10
X = X.iloc[:, :-3]
model = RandomForestRegressor(random_state=50).fit(X.drop(['ID', 'Take'], axis=1), y.MCE_Score)
s = pd.Series(index=X.drop(['ID', 'Take'], axis=1).columns, data=model.feature_importances_).sort_values(ascending=False)
X = X.loc[:, list(s.index[:N_FEATURES]) + ['ID', 'Take']]
print(X.head())
print(X.shape)

trues, trues_cont = [], []
predicts, predicts_cont = [], []

for id in np.unique(X.ID):
    for take in np.unique(X[X.ID == id].Take):
        curr_df = X[(X.ID == id) & (X.Take == take)].drop(['ID', 'Take'], axis=1)
        curr_y = y[(y.ID == id) & (y.Take == take)]
        trues.append(np.unique(pd.cut(curr_y.MCE_Score, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values()))[0])
        trues_cont.append(np.unique(curr_y.MCE_Score)[0])

        print(curr_df.shape, curr_y.shape)

        train_df = X.drop(['ID', 'Take'], axis=1).drop(curr_df.index, axis=0)
        y_train = y.drop(curr_y.index, axis=0).MCE_Score

        print(train_df.shape, y_train.shape)

        model = SVR().fit(train_df, y_train)
        predicts_cont.append(np.median(model.predict(curr_df)))
        preds = pd.cut(model.predict(curr_df), [0, 1, 2, 3, 4, 5], labels=MCE_categories.values())

        preds_dict = {'Prob_Insuficiente': round(sum(preds == 'Insuficiente')/len(preds), 2),
                      'Prob_Regular': round(sum(preds == 'Regular')/len(preds), 2),
                      'Prob_Bueno': round(sum(preds == 'Bueno')/len(preds), 2),
                      'Prob_Excelente': round(sum(preds == 'Excelente')/len(preds), 2)}
        print(trues[-1], list(preds_dict.values()), MCE_categories[int(np.argmax(list(preds_dict.values())) + 2)])

        predicts.append(MCE_categories[int(np.argmax(list(preds_dict.values())) + 2)])

print('****')
print('Preds:', predicts)
print('Trues:', trues)
print('****')
predicts = pd.Categorical(predicts, categories=list(MCE_categories.values()), ordered=True)
trues = pd.Categorical(trues, categories=list(MCE_categories.values()), ordered=True)

print(accuracy_score(trues, predicts))
print(r2_score(trues_cont, predicts_cont))
df = pd.DataFrame({'True_values': trues_cont, 'Predicted_values': predicts_cont}).sort_values(by='True_values',
                                                                                              ascending=True)
plt.scatter(x=range(len(trues_cont)), y=df['True_values'], c='b', label='True')
plt.scatter(x=range(len(predicts_cont)), y=df['Predicted_values'], c='r', label='Predicted')
plt.legend()
plt.show()

