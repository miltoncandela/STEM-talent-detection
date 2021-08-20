# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code explores both target variables categories (MCE, PSI), using a processed CSV file from the
# "processed" folder, created using the "createCSV.py" file, which joins the biometric devices and the predicted scores.
# It creates multiple visualization tools, in order to extract data insights and do a quick feature selection process.

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from shap import TreeExplainer, summary_plot
import pandas as pd
import seaborn as sns

from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler


def figure_generator(folder, scores_categories, sampling_method='None'):
    """
    This function generates five plots in order to perform an Exploratory Data Analysis (EDA):

    1. Bar plot on feature importance according to the GINI coefficient, computed by a Random Forest Classifier.
    2. Random Forest Classifier's confusion matrix on the testing dataset, using the top N_FEATURES features.
    3. Pair plot using the top N_FEATURES features on the testing dataset and true Y values.
    4. Pair plot using the top N_FEATURES features on the testing dataset and predicted Y values.
    5. SHAP summary plot to understand the model's predictions and relationship between source and target variables.

    :param string folder: Name of the folder where the plots would be stored, which is the same as the DataFrame column.
    :param dictionary scores_categories: Label encoding from numeric to categorical variable.
    :param sampling_method: Type of resampling method being used.
    """

    # Data Extraction #
    # The inverse categories are defined, as they are convenient to encode them into numerical features is convenient.
    inv_categories = {v: k for k, v in scores_categories.items()}

    # Target variables that are not currently used, would be dropped.
    x = pd.read_csv('processed/filtered_{}_Score_2.csv'.format(folder[:3])).drop(['ID', 'Take'], axis=1)
    y = x.pop(folder)
    x = x.iloc[:, :-3]
    y = y.astype('category')

    # Class imbalance #
    # There exists a class imbalance on both MCE and PSI categories, one category has lower observations than others,
    # "imblearn" is a library with techniques that reduce the class imbalance, one option is to under-sample the data,
    # leading to fewer observations and data loss, using for example RandomUnderSampler to randomly remove majority
    # class observations, until reached the number of least observations. While the other option is to up-sample the
    # data, using RandomUnderSampler or SMOTE (Synthetic Minority Oversampling Technique) to avoid data loss.
    sampling_dict = {'SMOTE': SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=0),
                     'Rand_Under': RandomUnderSampler(random_state=50),
                     'Rand_Over': RandomOverSampler(random_state=50)}

    x_resampled, y_resampled = x, y
    if sampling_method != 'None':
        x_resampled, y_resampled = sampling_dict[sampling_method].fit_resample(x, y)

    # Class imbalance is solved, and so accuracy can be used as a valid metric.
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, train_size=0.7, random_state=60)

    # Feature selection #
    # An initial Random Forest (RF) classifier is trained using all the features on the training dataset, it overfitts
    # the whole dataset, although it provides good accuracy and would serve to select the best features.
    model = RandomForestClassifier(random_state=50).fit(x_resampled, y_resampled)

    # The RF model computes the Gini importance index, which would be used as a feature selection method, to n features.
    s = pd.Series(index=x.columns, data=model.feature_importances_).sort_values(ascending=False)

    # The following bar plot visually establish the importance of the best features according to the RF model.
    emotions = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'angry']
    feat_eeg = ['Alpha', 'LowBeta', 'HighBeta', 'Gamma', 'Theta'] + ['Load', 'Fatigue', 'Engagement']

    def color_bar(feature):
        """
        This function would take the name of a feature, and then returns the assigned color depending on the device.

        :param string feature: Name of the feature, could be a combined feature, as splits using underscore "_" and
        hyphen "-" are being made to take the first feature (combined features are done on the same device).
        :return: A color in Hexadecimal that would be taken into the bar plot, to identify each feature's device.
        """

        feature = feature.split('_')[0].split('-')[0]
        return '#FACF5A' if feature in emotions else '#455D7A' if feature in feat_eeg else '#F95959'

    # A bar plot on feature importance according to the Gini index computed by the RF trained model, is generated.
    fig = plt.figure()
    plt.bar(s.index[:25], s[:25], color=[color_bar(col) for col in list(s.index)])
    plt.xlabel('Feature')
    plt.ylabel("Importance")
    plt.title('Top 25 most important features when predicting {} category'.format(folder[:3]))
    plt.xticks(rotation=90)
    plt.legend(handles=[mpatches.Patch(color='#FACF5A', label='Emotions'),
                        mpatches.Patch(color='#455D7A', label='EEG'),
                        mpatches.Patch(color='#F95959', label='Empatica')])
    fig.savefig('figures/{}/{}/top_features_bar_plot.png'.format(folder, sampling_method), bbox_inches='tight')

    # These best features are used as a filter on the DataFrames, declared on separate, filtered DataFrames.
    x_train_filter = x_train.loc[:, s.index[:N_FEATURES]]
    x_test_filter = x_test.loc[:, s.index[:N_FEATURES]]

    # Modeling #
    # Using the best features, a second model is then trained, this model performs lower than the previous model, it is
    # understandable, as it uses n features instead of the total features. Parsimonious models, which uses the least
    # amount of features, are ideal because it efficiently solves a complex task, with the least amount of computations.
    model = RandomForestClassifier(random_state=20).fit(x_train_filter, y_train)

    # The confusion matrix is ideal for multi-class modelling, as it gives insights on the difficult to model classes,
    # on which focus and more processing is required to completely disseminate every class.
    display = plot_confusion_matrix(model, x_test_filter, y_test)
    display.ax_.set_title("Model's confusion matrix using {} features".format(N_FEATURES))
    display.ax_.set_xticklabels((tuple(inv_categories.keys())))
    display.ax_.set_yticklabels((tuple(inv_categories.keys())))
    plt.gcf().savefig('figures/{}/{}/confusion_matrix.png'.format(folder, sampling_method), bbox_inches='tight')

    # Visualization #
    # Using seaborn, a pair plot can be done using a DataFrame, though, the number of columns in that DataFrame must be
    # small, as it is a matrix of plots and it can be unclear when the number of features is huge. These pair plots are
    # useful in filtered DataFrames, as data insights can be drawn, deepening the understanding on the target variable.

    # The following colors in hexadecimal format would be used to declare the class of each observation:
    # Blue = #0000FF
    # Green =  #00FF00
    # Yellow = #FFFF00
    # Orange = #FF8000
    # Red = #FF0000
    palette = ['#0000FF', '#00FF00', '#FF8000', '#FF0000'] if len(scores_categories.keys()) == 4 \
        else ['#00FF00', '#FF0000']

    # The whole dataset is filtered using the best features and a "Category" column in added, in order to used it as a
    # parameter in the seaborn pair plot, we define the shape of the markers depending on the number of targets.
    x_resampled = x_resampled.loc[:, s.index[:N_FEATURES]]
    x_resampled['Categoria'] = pd.Categorical(y_resampled, categories=list(reversed(scores_categories.values())))
    markers = ['s', 's', 'o', 'o'] if len(scores_categories.keys()) > 2 else ['s', 'o']
    fig = sns.pairplot(x_resampled, hue="Categoria", markers=markers, palette=palette)
    fig.savefig('figures/{}/{}/pair_plot.png'.format(folder, sampling_method),
                bbox_inches='tight', orientation='portrait')

    # The previous pair plot make reduced-size scatter plots, and so individual scatter plots of the pair plot's
    # features would also be made, first we remove all the figures in the folder.
    for file in os.listdir('figures/{}/{}/plots/'.format(folder, sampling_method)):
        os.remove('figures/{}/{}/plots/{}'.format(folder, sampling_method, file))

    # The following for loop iterates over all the best features used on the pair plot (5), and generates individual
    # plots to better visualize the interaction between the features and the impact on the target variable. Although
    # it has an if-statement to avoid plotting the same variable on the x-axis and y-axis (which is represented as a
    # density plot in the pair plot). The axis are limited to 0 and 1, because that is the domain of all the features,
    # due to the quantity of plots being generated, a separate folder names "plots" is used to save all the plots.
    for num_column_i, feature_i in enumerate(x_resampled.columns):
        for num_column_j, feature_j in enumerate(x_resampled.columns):
            if feature_i != feature_j and feature_i != 'Categoria' and feature_j != 'Categoria':
                fig = plt.figure()
                ax = fig.gca()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                sns.scatterplot(x=feature_i, y=feature_j, data=x_resampled, alpha=0.8, hue='Categoria', palette=palette)
                plt.xlabel(feature_i)
                plt.ylabel(feature_j)
                plt.title('{} and {} on {} category'.format(feature_i, feature_j, folder[:3]))
                plt.legend()
                fig.savefig('figures/{}/{}/plots/{}-{}.png'.format(folder, sampling_method,
                                                                   feature_i, feature_j), bbox_inches='tight')

    # Analysis #
    # A further SHapley Additive exPlanations (SHAP) analysis was done, which takes "black box" machine learning models
    # and computes relationships between the features and the target variable. A final RF is trained on a encoded,
    # numeric target variable, due to the progressive cardinality between categories, the SHAP plot would be useful
    # to detect whether the increase or decrease of a feature is related to an increase or decrease of our target class.
    x_resampled.drop('Categoria', axis=1, inplace=True)
    y_resampled = pd.Categorical(y_resampled, categories=list(scores_categories.values()), ordered=True)
    model = RandomForestClassifier(random_state=20).fit(x_resampled, y_resampled)
    shap_values = TreeExplainer(model).shap_values(x_resampled)[1]

    # The categories are ordered so tha the summary plot makes sense, the bar on the right represents the increment
    # of a feature value according to the feature order, and SHAP value is how far an observation is to their mean.
    fig = plt.figure()
    summary_plot(shap_values, x_resampled, show=False, class_names=list(scores_categories.values()))
    ax = fig.gca()
    ax.set_title("SHAP summary plot on predicting {}".format(folder[:3]))
    fig.savefig('figures/{}/{}/SHAP.png'.format(folder, sampling_method), format='png', dpi=150, bbox_inches="tight")
    print('Figures saved on figures/{}/{}/'.format(folder, sampling_method))


# The following lines removes a matplotlib warning that pops out when multiple figures are being generated.
mpl.rcParams['figure.max_open_warning'] = 0
# The usual categories are being created, and so combinations of the dictionaries and the name of the categorical
# variable are being composed. Some sampling methods are also used to test differences, although the filtered
# dataset obtained the best features based on a non-balanced dataset, using any of these sampling methods.
N_FEATURES = 5
MCE_categories = {2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
PSI_categories = {0: 'Negativo', 1: 'Positivo'}
combinations = [['MCE_Category', MCE_categories], ['PSI_Category', PSI_categories]]
sampling_methods = ['None', 'SMOTE', 'Rand_Under', 'Rand_Over']
for combination in combinations:
    for s_method in sampling_methods:
        figure_generator(combination[0], combination[1], s_method)
