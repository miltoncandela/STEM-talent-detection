import pandas as pd

X = pd.DataFrame({'Recamaras': [3, 2, 2, 3, 3, 3],
                  'Banos': [2, 1, 1, 2, 2, 2],
                  'Area': [93, 47, 70, 134, 100, 174],
                  'Precio': [1980000, 1500000, 1490000,
                             1895000, 1890000, 2570000]})

print(X)

# X.Precio = X.Precio
Y = X.pop('Precio')

print(X)
print(Y)

from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, Y)
print(model.coef_)
print(model.intercept_)

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import r2_score

df_results = pd.DataFrame({'True_values': Y,
                           'Predicted_values': Y - 10000}).sort_values(by='True_values', ascending=True)


fig = plt.figure()
plt.scatter(x=range(df_results.shape[0]), y=df_results['True_values']/1000000, c='b', label='Referencia')
plt.scatter(x=range(df_results.shape[0]), y=df_results['Predicted_values']/1000000, c='r', label='Predichos')
# plt.xticks(range(df_results.shape[0]), df_results.index)
a = plt.gca()
a.set_title('Predicciones ideales para el precio de una casa')
a.set_xlabel("Índice de casa, ordenado de menor de mayor")
a.set_ylabel('Precio de casa (millones)')
plt.legend(loc='upper left')
at = AnchoredText("Coeficiente de determinación $(R^2)$: {}".format(round(r2_score(Y, Y - 10000), 2)),
                  loc='lower right')
a.add_artist(at)
plt.show()
