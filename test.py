import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

metric = 'PSI_Score'

NOT_VALID_IDS = ['EJ01', 'ST02']
N_FEATURES = 20
x = pd.read_csv('processed/filtered_PSI_Score_2.csv', dtype={'MCE_Category': 'category', 'PSI_Category': 'category',
                                                      'ID': 'string', 'Take': int})
x = x.drop(x[x.ID.isin(NOT_VALID_IDS)].index, axis=0)

# The desired target variable is popped form the main DataFrame and information variables are removed from it.
y = x.pop(metric)
x = x.drop(['ID', 'Take'], axis=1).iloc[:, :-3].astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=50)

model = RandomForestRegressor(random_state=50).fit(x_train, y_train)
s = pd.Series(data=model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print(s)

x_train_1 = x_train.loc[:, list(s.index[:N_FEATURES])]
x_test_1 = x_test.loc[:, list(s.index[:N_FEATURES])]

model = RandomForestRegressor(random_state=50).fit(x_train_1, y_train)
print(r2_score(y_train, model.predict(x_train_1)))
print(r2_score(y_test, model.predict(x_test_1)))

model = RandomForestRegressor(random_state=50).fit(x, y)
s = pd.Series(data=model.feature_importances_, index=x.columns).sort_values(ascending=False)
print(s)

x = x.loc[:, list(s.index[:N_FEATURES])]
x_train_2 = x_train.loc[:, list(s.index[:N_FEATURES])]
x_test_2 = x_test.loc[:, list(s.index[:N_FEATURES])]

model = RandomForestRegressor(random_state=50).fit(x_train_2, y_train)
print(r2_score(y_train, model.predict(x_train_2)))
print(r2_score(y_test, model.predict(x_test_2)))

x['MCE_Score'] = y
x = x.sample(frac=1, random_state=30).reset_index(drop=True)
y = x.pop('MCE_Score')

cross_scores = np.abs(cross_val_score(RandomForestRegressor(random_state=50), x, y, cv=10, scoring='r2'))
print(cross_scores)
print('Cross-validation r_squared score using RF:', np.mean(cross_scores))
