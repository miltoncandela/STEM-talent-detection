import pandas as pd
import numpy as np
from pickle import load
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
X = pd.read_csv('processed/combined_df.csv', dtype={'MCE_Category': 'category', 'PSI_Category': 'category',
                                                    'ID': 'string', 'Take': int})
X.dropna(axis=0, how='any', inplace=True)
y = pd.DataFrame({'MCE_Score': X.pop('MCE_Score'), 'ID': X.ID, 'Take': X.Take})
print(y.head())
X = X.iloc[:, :-3]

model = load(open('processed/model.pkl', 'rb'))
best_columns = open('processed/MCE_Score_Cols.txt', 'r').read().split('\n')
X = X.loc[:, best_columns[:25] + ['ID', 'Take']]
print(X.head())

trues, trues_cont = [], []
predicts, predicts_cont = [], []

for id in np.unique(X.ID):
    for take in np.unique(X[X.ID == id].Take):
        curr_df = X[(X.ID == id) & (X.Take == take)].drop(['ID', 'Take'], axis=1)
        trues.append(np.unique(pd.cut(y[(y.ID == id) & (y.Take == take)].MCE_Score, [0, 1, 2, 3, 4, 5],
                                      labels=MCE_categories.values()))[0])
        trues_cont.append(np.unique(y[(y.ID == id) & (y.Take == take)].MCE_Score)[0])
        predicts_cont.append(np.mean(model.predict(curr_df)))
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
df = pd.DataFrame({'True_values': trues_cont, 'Predicted_values': predicts_cont})
plt.scatter(x=range(len(trues_cont)), y=df['True_values'], c='b', label='True')
plt.scatter(x=range(len(predicts_cont)), y=df['Predicted_values'], c='r', label='Predicted')
plt.legend()
plt.show()

