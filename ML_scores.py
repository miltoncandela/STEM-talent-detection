import pandas as pd
import numpy as np

categorias = {'Malo' : 1,
              'Insuficiente' : 2,
              'Regular' : 3,
              'Bueno' : 4,
              'Excelente' : 5}
inv_cat = {v: k for k, v in categorias.items()}
X = pd.read_csv('ResultadosMCE.csv')
X.drop('Score', axis = 1, inplace = True)
Y = X.pop('Categoria')
#Y = pd.Series([categorias[x] for x in Y]).astype('category')

from collections import Counter
print(sorted(Counter(Y).items()))

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
rus = RandomOverSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, Y)
print(sorted(Counter(y_resampled).items()))

p = 0.7

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = p, random_state = 60)

from sklearn.decomposition import PCA
#pca = PCA(3).fit(X_train)
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#model = SVC(kernel = 'rbf').fit(X_train, y_train)
model = RandomForestClassifier(random_state=50).fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, model.predict(X_test)))
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

s = pd.Series(index = X.columns, data = model.feature_importances_).sort_values(ascending = False)
n = 5

import matplotlib.pyplot as plt
plt.bar(s.index[:n],s[:n])
plt.xlabel('Feature')
plt.ylabel("Importance")
plt.title('Feature importance on predicting class, using {} features'.format(n))
plt.xticks(rotation=90)
plt.show()

X_train_filt = X_train.loc[:,s.index[:n]]
X_test_filt = X_test.loc[:,s.index[:n]]

model = RandomForestClassifier(random_state=25).fit(X_train_filt, y_train)
print(accuracy_score(y_train, model.predict(X_train_filt)))
print(accuracy_score(y_test, model.predict(X_test_filt)))
y_pred = model.predict(X_test_filt)
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(model, X_test_filt, y_test)
disp.ax_.set_title("Model's confusion matrix using {} features".format(n))
disp.ax_.set_xticklabels((tuple(categorias.keys())[1:]))
disp.ax_.set_yticklabels((tuple(categorias.keys())[1:]))
plt.show()

y_test.index = range(y_test.shape[0])
X_test_filt.index = range(X_test_filt.shape[0])
X_test_filt['Categoria'] = y_test

import seaborn as sns
sns.pairplot(X_test_filt, hue="Categoria")
plt.show()

'''
y_test.index = range(y_test.shape[0])
X_test_filt.index = range(X_test_filt.shape[0])
#colores = ['b', 'g', 'r', 'c']
#colores = dict(zip(range(2,6), ['b', 'g', 'r', 'c']))
colores = dict(zip(list(categorias.keys())[1:], ['b', 'g', 'r', 'c']))
for label in np.unique(y_test):
    idx = y_test[y_test == label].index
    plt.scatter(X_test_filt.iloc[idx,1], X_test_filt.iloc[idx,0],
                c = [colores[x] for x in y_test[y_test == label]],
                label = label, alpha = 0.3)
#plt.scatter(X_test_filt.iloc[:,0], X_test_filt.iloc[:,1])
plt.xlabel(X_test_filt.columns[1])
plt.ylabel(X_test_filt.columns[0])
plt.title('{} vs {}, most important features'.format(X_test_filt.columns[1],
                                                     X_test_filt.columns[0]))
plt.legend()
plt.show()
'''

#from shap import TreeExplainer
#from shap import summary_plot
#shap_values = TreeExplainer(model).shap_values(X_test)[1]
#f = summary_plot(shap_values, X_test)
#f.savefig('SHAP.pdf', bbox_inches='tight')