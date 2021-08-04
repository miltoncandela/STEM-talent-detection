# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code explores the "score" category, using a processed CSV file from the "processed" folder, created
# using the "createCSV.py" file, which joins the biometric devices and the predicted scores.


# Data Extraction #
import pandas as pd

# The MCE categories are defined, as they would be explored and encoding, as well as decoding, is convenient.
MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
inv_categories = {v: k for k, v in MCE_categories.items()}

# Target variables that are not currently used, would be dropped.
X = pd.read_csv('processed/combined_df.csv')
X.drop(['MCE_Score', 'PSI_Score', 'PSI_Category'], axis=1, inplace=True)
Y = X.pop('MCE_Category')


# Class imbalance #
from collections import Counter

print(sorted(Counter(Y).items()))
# [('Bueno', 118), ('Excelente', 114), ('Insuficiente', 22), ('Regular', 169)]
# It exists a class imbalance on this category, as "Insuficiente" has lower observations that the other categories,
# "imblearn" is a library with techniques that reduce the class imbalance, one option is to under-sample the data,
# leading to fewer observations and data loss, using for example RandomUnderSampler to randomly remove majority class
# observations, until reached the number of least observations. While the other option is to up-sample the data,
# using RandomUnderSampler or SMOTE (Synthetic Minority Oversampling Technique) to avoid data loss.

from sklearn.model_selection import train_test_split
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=0).fit_resample(X, Y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.7, random_state=60)
print(sorted(Counter(y_resampled).items()))
# [('Bueno', 169), ('Excelente', 169), ('Insuficiente', 169), ('Regular', 169)]
# As it can be seen, class imbalance is solved, and so accuracy can be used as a valid metric.


# Feature selection #
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# An initial Random Forest (RF) classifier is trained using all the features on the training dataset, it overfitts
# the training dataset, although it provides 0.89 accuracy on the testing dataset.
model = RandomForestClassifier(random_state=50).fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))

# This RF model computes the GINI importance index, which would be used as a feature selection method, to n features
s = pd.Series(index=X.columns, data=model.feature_importances_).sort_values(ascending=False)
n_features = 5

# The following bar plot visually established the importance of the best features according to the RF model.
import matplotlib.pyplot as plt
plt.bar(s.index[:n_features], s[:n_features])
plt.xlabel('Feature')
plt.ylabel("Importance")
plt.title('Feature importance on predicting class, top {} features'.format(n_features))
plt.xticks(rotation=90)
plt.show()

# These best features are used as a filter on the DataFrames, declared on separate, filtered DataFrames
X_train_filter = X_train.loc[:, s.index[:n_features]]
X_test_filter = X_test.loc[:, s.index[:n_features]]


# Modeling #
# Using the best features, a second model is then trained, this model performs lower than the previous model, it is
# understandable, as it uses n features instead of the total 104 features. Models that use the least amount of features
# or parsimonious models, are ideal because it efficiently solves a complex task, with the least amount of computations.
model = RandomForestClassifier(random_state=20).fit(X_train_filter, y_train)
print(accuracy_score(y_train, model.predict(X_train_filter)))
print(accuracy_score(y_test, model.predict(X_test_filter)))
print(confusion_matrix(y_test, model.predict(X_test_filter)))

# The confusion matrix is ideal for multi-class modelling, as it gives insights on the difficult to model classes,
# on which focus and more processing is required to completely disseminate every class.
from sklearn.metrics import plot_confusion_matrix
display = plot_confusion_matrix(model, X_test_filter, y_test)
display.ax_.set_title("Model's confusion matrix using {} features".format(n_features))
display.ax_.set_xticklabels((tuple(inv_categories.keys())[1:]))
display.ax_.set_yticklabels((tuple(inv_categories.keys())[1:]))
plt.show()


# Visualization #
# Each index is reset, as it was random sampled from the initial DataFrame
y_test.index = range(y_test.shape[0])
X_test_filter.index = range(X_test_filter.shape[0])
X_test_filter['Categoria'] = pd.Categorical(y_test)

# Using seaborn, a pair plot can be done using a DataFrame, it is important that the number of columns in that DataFrame
# is not too big, as it is a matrix of plots and it can be unclear when the number of features is huge. These pair plots
# are useful in filtered DataFrames, as data insights can be drawn, deepening the understanding of the target variable.
import seaborn as sns
sns.pairplot(X_test_filter, hue="Categoria", markers=['o', 's',])
X_test_filter.drop('Categoria', axis=1, inplace=True)
plt.show()

# A further SHapley Additive exPlanations (SHAP) analysis was done, which uses "black box" machine learning models
# and computes relationships between the features and the target variable. A final RF is trained on a encoded, target
# variable, because of the increasing category of our target classes (4 is better than 1), the SHAP plot would be useful
# to detect whether the increase or decrease of a feature is related to the increase or decrease of our target class.
from shap import TreeExplainer, summary_plot

y_train_enc = pd.Series([inv_categories[str(x)] for x in y_train]).astype('category')
model = RandomForestClassifier(random_state=20).fit(X_train_filter, y_train_enc)
shap_values = TreeExplainer(model).shap_values(X_test_filter)[1]
summary_plot(shap_values, X_test_filter)
