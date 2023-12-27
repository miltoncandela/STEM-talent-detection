# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code evaluates and creates a Machine Learning model based on the continuous target variable and its
# filtered dataset, created by "filterCSV.py". It uses a Random Forest (RF) regression model using a constant 75:25
# split when creating the training and testing dataset. This script in particular prints the results of the evaluation
# rather than creating a plot, this is due to the fact that the best parameters such as the number of features
# and the split are being used, although any of the models are optimized, as their default parameters are being used.

import numpy as np
import pandas as pd
from pickle import dump
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from sklearn.linear_model import LinearRegression
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def model_generation(metric, imbalance_solver=False):
    """
    This function evaluates and creates a Machine Learning regressor model, based on the given metric and whether
    a random under sampler is being used to solve class imbalance. Although, some classes are very scarce
    ("Insuficiente"), and so the results could be not representative of the whole population. Also, the feature
    selection process was done using the unbalanced dataset, and so features may not be optimal.

    :param string metric: Either PSI_Score or MCE_Score, depending on the preferred metric.
    :param bool imbalance_solver: Used to solve the class imbalance and represent balanced accuracies.
    """

    # The filtered dataset is read as a pandas DataFrame, a RandomUnderSampler is also applied depending on
    # whether the imbalance_solver parameter is true or false. Metric is popped and filtered to N_FEATURES.
    x = pd.read_csv('processed/filtered_{}_2.csv'.format(metric)).drop(['ID', 'Take'], axis=1)
    if imbalance_solver:
        x, y = RandomUnderSampler(random_state=50).fit_resample(x, x['{}_Category'.format(metric[:3])])
    y = x.pop(metric)
    x = x.iloc[:, :-3].iloc[:, :N_FEATURES]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=50)

    # As it was said, the r_squared metrics would be printed rather than plotted into a plot of some kind. Here three
    # types of ML regression models are used: Linear Regression as baseline, while Gradient Boosting Regressor (GBM) and
    # Random Forest (RF) to test the best performance and export as the best model
    print('*****')
    print('r_squared scores using various models')

    model = LinearRegression().fit(x_train, y_train)
    print('Linear regression:', r2_score(y_test, model.predict(x_test)))

    model = GradientBoostingRegressor(random_state=50).fit(x_train, y_train)
    print('Gradient boosting regressor:', r2_score(y_test, model.predict(x_test)))

    model = RandomForestRegressor(random_state=50).fit(x_train, y_train)
    print('Random forest regressor:', r2_score(y_test, model.predict(x_test)))

    # Cross-validation metric using 10-folds is also employed, in this case the dataset must be previously shuffled
    # and so because the train_test_split function is not used to shuffle the data, it needs to be done manually.
    x['MCE_Score'] = y
    x = x.sample(frac=1, random_state=30).reset_index(drop=True)
    y = x.pop('MCE_Score')

    # Considering 13 students are being used, a cross-validation using 10-folds would be using 77% of the data for
    # training and 23% for testing. And so it resembles the train/test split that was previously done to train
    # the past models, using these parameters, the score should be similar to the one displayed by the previous model.
    cross_scores = np.abs(cross_val_score(RandomForestRegressor(random_state=50), x, y, cv=10, scoring='r2'))
    print('Cross-validation r_squared score using RF:', np.mean(cross_scores))

    # Accuracy is also obtained to test whether the continuous predictions make sense when encoded to their categorical
    # value. And so a conditional statement is employed based on whether metric is MCE_Score or PSI_Score.
    trues = pd.cut(y_test, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values()) if metric == 'MCE_Score' \
        else ['Positivo' if psi_score > 0 else 'Negativo' for psi_score in y_test]
    preds = pd.cut(model.predict(x_test), [0, 1, 2, 3, 4, 5], labels=MCE_categories.values()) if metric == 'MCE_Score' \
        else ['Positivo' if psi_score > 0 else 'Negativo' for psi_score in model.predict(x_test)]

    # Predictions and reference values are concatenated into a DataFrame to sort from lower to greater.
    df = pd.DataFrame({'True_values': y_test, 'Predicted_values': model.predict(x_test)}).sort_values(by='True_values',
                                                                                                      ascending=True)

    # The following lines of code create a scatter plot of predictions and reference values, using the previously
    # created DataFrame, it also change its labels and title depending on the metric that was chosen. Furthermore,
    # the accuracy and r2 score are also included in the plot using AnchoredText by matplotlib.
    fig = plt.figure()
    plt.scatter(x=range(x_test.shape[0]), y=df['True_values'], c='b', label='Reference')
    plt.scatter(x=range(x_test.shape[0]), y=df['Predicted_values'], c='r', label='Predicted')
    a = plt.gca()
    title_name = 'STEM interest' if metric == 'PSI_Score' else 'performance'
    a.set_title('Reference and predicted {} using RF with 17 features'.format(title_name))
    a.set_xlabel("Sample's index using testing dataset from 80:20 split")
    y_name = 'Delta of STEM interest, given pre and pos evaluation' if metric == 'PSI_Score' else \
        "Student's performance using manual algorithm"
    a.set_ylabel(y_name)
    plt.legend(loc='upper left')
    at = AnchoredText("Accuracy on predicting classes: {}%\nCoefficient of determination $(R^2)$: {}".format(
        round(accuracy_score(trues, preds)*100, 2), round(r2_score(y_test, model.predict(x_test)), 2)),
        loc='lower right')
    a.add_artist(at)
    fig.savefig('figures/results/{}_prediction.png'.format(metric), bbox_inches='tight')

    # A confusion matrix is printed, here the function plot_confusion_matrix could not be used, as categorical
    # values were drawn from our initial continous-variable predictions.
    print('Confusion matrix:')
    print(confusion_matrix(trues, preds))
    print('Accuracy:', accuracy_score(trues, preds))
    print('Number of samples for each category:')
    print('True:', Counter(trues))
    print('Predicted:', Counter(preds))

    # The best model (RF) is exported as a pickle file on the "processed" folder, further used to predict on new data.
    # dump(model, open('processed/{}_model.pkl'.format(metric), 'wb'))


# As on previous functions, the dictionary of MCE_Categories is established to encode continuous variables.
MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
N_FEATURES = 20
for score in ['PSI_Score', 'MCE_Score']:
    model_generation(score)
