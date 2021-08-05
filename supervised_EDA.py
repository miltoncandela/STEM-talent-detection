# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code explores both target variables categories (MCE, PSI), using a processed CSV file from the
# "processed" folder, created using the "createCSV.py" file, which joins the biometric devices and the predicted scores.
# It creates multiple visualization tools, in order to extract data insights and do a quick feature selection process.
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.max_open_warning'] = 0

from shap import TreeExplainer, summary_plot
import pandas as pd
import seaborn as sns

from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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
    x = pd.read_csv('processed/combined_df.csv')
    y = x.pop(folder)
    x = x.iloc[:, :-3]
    y = y.astype('category')

    # Class imbalance #
    from collections import Counter

    # print(sorted(Counter(y).items()))
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
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, train_size=0.7, random_state=60)
    # Class imbalance is solved, and so accuracy can be used as a valid metric.

    # Feature selection #
    # An initial Random Forest (RF) classifier is trained using all the features on the training dataset, it overfitts
    # the training dataset, although it provides 0.89 accuracy on the testing dataset.
    model = RandomForestClassifier(random_state=50).fit(x_resampled, y_resampled)

    # This RF model computes the GINI importance index, which would be used as a feature selection method, to n features
    s = pd.Series(index=x.columns, data=model.feature_importances_).sort_values(ascending=False)

    # The following bar plot visually established the importance of the best features according to the RF model.
    fig = plt.figure()
    plt.bar(s.index[:25], s[:25])
    plt.xlabel('Feature')
    plt.ylabel("Importance")
    plt.title('Top 25 most important features when predicting {} category'.format(folder[:3]))
    plt.xticks(rotation=90)
    fig.savefig('figures/{}/{}/top_features_bar_plot.png'.format(folder, sampling_method), bbox_inches='tight')

    # These best features are used as a filter on the DataFrames, declared on separate, filtered DataFrames
    x_train_filter = x_train.loc[:, s.index[:N_FEATURES]]
    x_test_filter = x_test.loc[:, s.index[:N_FEATURES]]

    # Modeling #
    # Using the best features, a second model is then trained, this model performs lower than the previous model, it is
    # understandable, as it uses n features instead of the total 104 features. Parsimonious models, which uses the least
    # amount of features, are ideal because it efficiently solves a complex task, with the least amount of computations.
    model = RandomForestClassifier(random_state=20).fit(x_train_filter, y_train)
    # print(accuracy_score(y_train, model.predict(x_train_filter)), accuracy_score(y_test, model.predict(x_test_filter)))

    # The confusion matrix is ideal for multi-class modelling, as it gives insights on the difficult to model classes,
    # on which focus and more processing is required to completely disseminate every class.
    display = plot_confusion_matrix(model, x_test_filter, y_test)
    display.ax_.set_title("Model's confusion matrix using {} features".format(N_FEATURES))
    display.ax_.set_xticklabels((tuple(inv_categories.keys())))
    display.ax_.set_yticklabels((tuple(inv_categories.keys())))
    plt.gcf().savefig('figures/{}/{}/confusion_matrix.png'.format(folder, sampling_method), bbox_inches='tight')

    # Visualization #
    # Using seaborn, a pair plot can be done using a DataFrame, it is the number of columns in that DataFrame have to be
    # small, as it is a matrix of plots and it can be unclear when the number of features is huge. These pair plots are
    # useful in filtered DataFrames, as data insights can be drawn, deepening the understanding on the target variable.
    x_resampled = x_resampled.loc[:, s.index[:N_FEATURES]]
    x_resampled['Categoria'] = pd.Categorical(y_resampled)
    fig = sns.pairplot(x_resampled, hue="Categoria", markers=['o', 's', 'D', 'H'][:len(set(y))])
    fig.savefig('figures/{}/{}/pair_plot.png'.format(folder, sampling_method),
                bbox_inches='tight', orientation='portrait')


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#1f77b4', '#ff7f0e']
    markers = ['o', 's', 'D', 'H']
    for num_column_i, feature_i in enumerate(x_resampled.columns):
        for num_column_j, feature_j in enumerate(x_resampled.columns):
            if feature_i != feature_j and feature_i != 'Categoria' and feature_j != 'Categoria':
                fig = plt.figure()
                ax = fig.gca()

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                if feature_i == 'surprise':
                    ax.set_xlim(0, 0.2)
                if feature_j == 'surprise':
                    ax.set_ylim(0, 0.2)
                for i, curr_label in enumerate(sorted(list(set(y_resampled)), reverse=True)):
                    plt.scatter(x_resampled[x_resampled.Categoria == curr_label][feature_i],
                                x_resampled[x_resampled.Categoria == curr_label][feature_j],
                                label=str(curr_label), color=colors[inv_categories[curr_label]])
                plt.xlabel(feature_i)
                plt.ylabel(feature_j)
                plt.title('{} and {} on {} category'.format(feature_i, feature_j, folder[:3]))
                plt.legend()
                fig.savefig('figures/{}/{}/plots/{}-{}.png'.format(folder, sampling_method,
                                                                feature_i, feature_j), bbox_inches='tight')
    # A further SHapley Additive exPlanations (SHAP) analysis was done, which uses "black box" machine learning models
    # and computes relationships between the features and the target variable. A final RF is trained on a encoded,
    # numeric target variable, due to the progressive cardinality between categories, the SHAP plot would be useful
    # to detect whether the increase or decrease of a feature is related to an increase or decrease of our target class.

    x_resampled.drop('Categoria', axis=1, inplace=True)
    y_enc = pd.Series([inv_categories[str(x)] for x in y_resampled]).astype('category')
    model = RandomForestClassifier(random_state=20).fit(x_resampled, y_enc)
    shap_values = TreeExplainer(model).shap_values(x_resampled)[1]
    fig = plt.figure()
    summary_plot(shap_values, x_resampled, show=False, class_names=scores_categories.values)
    ax = fig.gca()
    ax.set_title("SHAP summary plot on predicting {}".format(folder[:3]))
    fig.savefig('figures/{}/{}/SHAP.png'.format(folder, sampling_method), format='png', dpi=150, bbox_inches="tight")
    print('Figures saved on figures/{}/{}/'.format(folder, sampling_method))


N_FEATURES = 5
MCE_categories = {2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
PSI_categories = {0: 'Negativo', 1: 'Positivo'}
combinations = [['MCE_Category', MCE_categories], ['PSI_Category', PSI_categories]]
sampling_methods = ['None', 'SMOTE', 'Rand_Under', 'Rand_Over']
for combination in combinations:
    for s_method in sampling_methods:
        figure_generator(combination[0], combination[1], s_method)
