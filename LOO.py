# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code evaluates and creates a Machine Learning model based on the continuous target variable and its
# filtered dataset, created by "filterCSV.py". It uses three machine learning models:
#
# 1. Multivariate Linear Regression (MLR)
# 2. Gradient Boosting Regressor (GBR)
# 3. Random Forest Regressor (RF)
#
# The validation used involves a Leave-One-Out approach for each class that the student took, and so both ID and Take
# information variables would be used to create multiple models removing a take, training the model using the other
# takes and finally validating the model using the removed take. This script in particular prints the results of the
# evaluation rather than creating a plot, this is due to the fact that the best parameters, like the number of features,
# and the split are being used, although any of the models are optimized, as their default parameters are being used.

import numpy as np
import pandas as pd
from pickle import dump
from collections import Counter
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Sin One Hot
# MCE = 0.13 0.37 Mean both
# PSI = 0.43 0.67 Mean both

# Con One Hot
# MCE = 0.39 0.60 Mean both
# PSI = 0.15 0.41 Mean both

MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}


def leave_one_out(metric, imbalance_solver=False):
    """
    The following function implements a LOO approach and prints a lot of useful information to measure a Machine
    Learning model on the processed dataset, given a certain target variable. It prints the following metrics:
    1. Pearson correlation and p-value for each feature and the given metric.
    2. R squared (R2) for each ML prediction and the true continuous value.
    3. Confusion matrix of predictions and true values of class metric.
    4. Accuracy and counter of classes of the true and predicted values.
    Additionally, it saves a scatter plot of sorted reference values and their predictions using the best model
    according to the target variable, in the "figures/results" folder.

    :param string metric: Either PSI_Score or MCE_Score, the target variable from which the figures and the model
     would be created.
    :param bool imbalance_solver: Whether the class imbalance problem is solved, using a Random Under Sampler, not
     recommended for MCE_Score as “Insuficiente” has a few takes, it might be used for a more accurate, accuracy class.
    """

    print('***** {} *****'.format(metric))
    x = pd.read_csv('processed/filtered_{}.csv'.format(metric))
    if imbalance_solver:
        x, y = RandomUnderSampler(random_state=50).fit_resample(x, x['{}_Category'.format(metric[:3])])
    y = pd.DataFrame({'Metric': x.pop(metric), 'ID': x.ID, 'Take': x.Take})
    x = x.iloc[:, :-3].loc[:, list(x.columns[:N_FEATURES]) + ['ID', 'Take']]
    if metric != 'MCE_Score':
        x = pd.concat([x, pd.DataFrame(OneHotEncoder().fit_transform(np.array(x.Take).reshape(-1, 1)).toarray())],
                      axis=1)

    scores_pearson = [np.abs(pearsonr(y['Metric'], x[feature])[0]) for feature in x.columns[:N_FEATURES]]
    p_value = [np.abs(pearsonr(y['Metric'], x[feature])[1]) for feature in x.columns[:N_FEATURES]]
    df_correlations = pd.DataFrame({'Feature': x.columns[:N_FEATURES], 'Correlation': scores_pearson, 'P': p_value})
    df_correlations = (df_correlations[df_correlations.P < 0.05].sort_values('Correlation', ascending=False).round(6)
                       .reset_index(drop=True))
    print(df_correlations.head(df_correlations.shape[0]))

    emotions = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'angry']
    feat_eeg = ['Alpha', 'LowBeta', 'HighBeta', 'Gamma', 'Theta'] + ['Load', 'Fatigue', 'Engagement']

    def color_bar(feature):
        """
        This function would take the name of a feature, and then returns the assigned color depending on the device.

        :param string feature: Name of the feature, could be a combined feature, as splits using underscore "_" and
        hyphen "-" are being made to take the first feature (combined features are done on the same device).
        :return string: A color in Hexadecimal that would be taken into the bar plot, to identify each feature's device.
        """

        feature = feature.split('_')[0].split('-')[0]
        return '#FACF5A' if feature in emotions else '#455D7A' if feature in feat_eeg else '#F95959'

    # A bar plot on feature importance according to the Gini index computed by the RF trained model, is generated.
    fig = plt.figure()
    plt.bar(df_correlations.Feature, df_correlations.Correlation,
            color=[color_bar(col) for col in list(df_correlations.Feature)])
    plt.xlabel('Características estadísticamente aceptadas')
    plt.ylabel("Coefficiente de correlación de Pearson")
    plt.title('Top {} características más relacionadas con el puntaje de {}'.format(df_correlations.shape[0], metric[:3]))
    plt.xticks(rotation=90)
    plt.legend(handles=[mpatches.Patch(color='#FACF5A', label='Emociones'),
                        mpatches.Patch(color='#455D7A', label='EEG'),
                        mpatches.Patch(color='#F95959', label='Empatica')])
    fig.savefig('figures/results/{}_top_correlated_features_bar_plot.png'.format(metric), bbox_inches='tight')

    trues_num, predicts_num_li, predicts_num_gbm, predicts_num_rf = [], [], [], []

    for cid in np.unique(x.ID):
        for take in np.unique(x[x.ID == cid].Take):
            curr_df = x[(x.ID == cid) & (x.Take == take)].drop(['ID', 'Take'], axis=1)
            curr_y = y[(y.ID == cid) & (y.Take == take)]

            trues_num.append(curr_y.reset_index().loc[0, 'Metric'])

            train_df = x.drop(['ID', 'Take'], axis=1).drop(curr_df.index, axis=0)
            y_train = y.drop(curr_y.index, axis=0).Metric

            predicts_num_li.append(np.round(np.mean(LinearRegression()
                                                    .fit(train_df, y_train).predict(curr_df)), 2))
            predicts_num_gbm.append(np.round(np.mean(GradientBoostingRegressor(random_state=50)
                                                     .fit(train_df, y_train).predict(curr_df)), 2))
            predicts_num_rf.append(np.round(np.mean(RandomForestRegressor(random_state=50)
                                                    .fit(train_df, y_train).predict(curr_df)), 2))

    print('Linear Regression:', r2_score(trues_num, predicts_num_li))
    print('Gradient boosted regressor:', r2_score(trues_num, predicts_num_gbm))
    print('Random forest regressor:', r2_score(trues_num, predicts_num_rf))

    predicts_num = predicts_num_rf if metric == 'PSI_Score' else predicts_num_gbm

    trues_cat = pd.cut(trues_num, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values()) if metric == 'MCE_Score' \
        else ['Positivo' if psi_score > 0 else 'Negativo' for psi_score in trues_num]
    predicts_cat = pd.cut(predicts_num, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values()) if metric == 'MCE_Score' \
        else ['Positivo' if psi_score > 0 else 'Negativo' for psi_score in predicts_num]

    df_results = pd.DataFrame({'True_values': trues_num,
                               'Predicted_values': predicts_num}).sort_values(by='True_values', ascending=True)

    # The following lines of code create a scatter plot of predictions and reference values, using the previously
    # created DataFrame, it also change its labels and title depending on the metric that was chosen. Furthermore,
    # the accuracy and r2 score are also included in the plot using AnchoredText by matplotlib.
    fig = plt.figure()
    plt.scatter(x=range(df_results.shape[0]), y=df_results['True_values'], c='b', label='Referencia')
    plt.scatter(x=range(df_results.shape[0]), y=df_results['Predicted_values'], c='r', label='Predicho')
    a = plt.gca()
    # title_name = 'STEM interest' if metric == 'PSI_Score' else 'performance'
    title_name = metric[:3]
    model_used = 'RF' if metric == 'PSI_Score' else 'GBR'
    a.set_title('Predicciones para el puntaje {} usando {} con {} características'.format(title_name, model_used, N_FEATURES))
    a.set_xlabel("Número de toma, ordenado de menor a mayor de acuerdo con el puntaje")
    y_name = 'Cambio en el interés psicométrico (después - antes)' if metric == 'PSI_Score' else \
        "Desempeño del estudiante usando el algoritmo manual"
    a.set_ylabel(y_name)
    plt.legend(loc='upper left')
    at = AnchoredText("Precisión al predecir categorías: {}%\nCoefficiente de determinación $(R^2)$: {}".format(
        round(accuracy_score(trues_cat, predicts_cat) * 100, 2), round(r2_score(trues_num, predicts_num), 2)),
        loc='lower right')
    a.add_artist(at)
    fig.savefig('figures/results/{}_prediction.png'.format(metric), bbox_inches='tight')

    print('Confusion matrix:')
    print(confusion_matrix(trues_cat, predicts_cat))
    print('Accuracy:', round(accuracy_score(trues_cat, predicts_cat) * 100, 2))
    print('True samples:', Counter(trues_cat))
    print('Predicted samples:', Counter(predicts_cat))

    x[metric] = y.Metric
    x = x.drop(['Take', 'ID'], axis=1).sample(frac=1, random_state=30).reset_index(drop=True)
    y = x.pop(metric)

    model = RandomForestRegressor(random_state=50).fit(x, y) if metric == 'PSI_Score' else LinearRegression().fit(x, y)
    dump(model, open('processed/{}_model.pkl'.format(metric), 'wb'))


N_FEATURES = 17
for target_variable in ['MCE_Score', 'PSI_Score']:
    leave_one_out(target_variable)
