import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from collections import Counter
from statistics import mode
from sklearn.preprocessing import OneHotEncoder
from matplotlib.offsetbox import AnchoredText
from scipy.stats.stats import pearsonr

# Sin One Hot
# MCE = 0.13 0.37 Mean both
# PSI = 0.43 0.67 Mean both

# Con One Hot
# MCE = 0.33 0.60 Mean both
# PSI = 0.15 0.41 Mean both

MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}


def leave_one_out(metric, imbalance_solver=False):

    print('***** {} *****'.format(metric))
    x = pd.read_csv('processed/filtered_{}_2.csv'.format(metric))
    y = pd.DataFrame({'Metric': x.pop(metric), 'ID': x.ID, 'Take': x.Take})
    x = x.iloc[:, :-3].loc[:, list(x.columns[:N_FEATURES]) + ['ID', 'Take']]
    if metric == 'MCE_Score':
        x = pd.concat([x, pd.DataFrame(OneHotEncoder().fit_transform(np.array(x.Take).reshape(-1, 1)).toarray())],
                      axis=1)

    scores_pearson = [np.abs(pearsonr(y['Metric'], x[feature])[0]) for feature in x.columns[:N_FEATURES]]
    p_value = [np.abs(pearsonr(y['Metric'], x[feature])[1]) for feature in x.columns[:N_FEATURES]]
    df_correlations = pd.DataFrame({'Feature': x.columns[:N_FEATURES], 'Correlation': scores_pearson, 'P': p_value})\
        .sort_values('Correlation', ascending=False).round(6).reset_index(drop=True)
    print(df_correlations.head(df_correlations.shape[0]))

    trues_num, predicts_num_li, predicts_num_gbm, predicts_num_rf = [], [], [], []

    for id in np.unique(x.ID):
        for take in np.unique(x[x.ID == id].Take):
            curr_df = x[(x.ID == id) & (x.Take == take)].drop(['ID', 'Take'], axis=1)
            curr_y = y[(y.ID == id) & (x.Take == take)]

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

    predicts_num = predicts_num_rf if metric == 'PSI_Score' else predicts_num_li

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
    plt.scatter(x=range(df_results.shape[0]), y=df_results['True_values'], c='b', label='Reference')
    plt.scatter(x=range(df_results.shape[0]), y=df_results['Predicted_values'], c='r', label='Predicted')
    a = plt.gca()
    title_name = 'STEM interest' if metric == 'PSI_Score' else 'performance'
    model_used = 'RF' if metric == 'PSI_Score' else 'LI'
    a.set_title('Reference and predicted {} using {} with {} features'.format(title_name, model_used, N_FEATURES))
    a.set_xlabel("Sample's index using leave-one-out validation")
    y_name = 'Delta of STEM interest, given pre and pos evaluation' if metric == 'PSI_Score' else \
        "Student's performance using manual algorithm"
    a.set_ylabel(y_name)
    plt.legend(loc='upper left')
    at = AnchoredText("Accuracy on predicting classes: {}%\nCoefficient of determination $(R^2)$: {}".format(
        round(accuracy_score(trues_cat, predicts_cat) * 100, 2), round(r2_score(trues_num, predicts_num), 2)),
        loc='lower right')
    a.add_artist(at)
    fig.savefig('figures/results/{}_prediction.png'.format(metric), bbox_inches='tight')

    print('Confusion matrix:')
    print(confusion_matrix(trues_cat, predicts_cat))
    print('Accuracy:', round(accuracy_score(trues_cat, predicts_cat) * 100, 2))
    print('True samples:', Counter(trues_cat))
    print('Predicted samples:', Counter(predicts_cat))


N_FEATURES = 17
for target_variable in ['MCE_Score', 'PSI_Score']:
    leave_one_out(target_variable)
