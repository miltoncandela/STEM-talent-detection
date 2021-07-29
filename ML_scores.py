import pandas as pd
import numpy as np

X = pd.read_csv('ResultadosMCE.csv')
X.drop('Score', axis = 1, inplace = True)
Y = X.pop('Categoria')

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
from sklearn.metrics import classification_report

#model = SVC(kernel = 'rbf').fit(X_train, y_train)
model = RandomForestClassifier(random_state=50).fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, model.predict(X_test)))
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

s = pd.Series(index = X.columns, data = model.feature_importances_).sort_values(ascending = False)
print(s)

#from shap import TreeExplainer
#from shap import summary_plot
#shap_values = TreeExplainer(model).shap_values(X_test)[1]
#f = summary_plot(shap_values, X_test)
#f.savefig('SHAP.pdf', bbox_inches='tight')