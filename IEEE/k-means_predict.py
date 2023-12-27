# Author: Milton Candela (https://github.com/milkbacon)
# Date: July 2021

# The following code takes the created model in "k-means.py" and predicts the class on unlabeled data. These unlabeled
# data would be our known classes, such as the MCE or PSI score, although it could be used for any information variable,
# in order to determine the usefulness for the current division of classes.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pickle import load

# The model and the XY ranges are loaded from the "processed" folder, these were created using "k-means.py".
model = load(open('processed/k-means_model.pkl', 'rb'))
file = open('processed/PCA_dimensions.txt', 'r')
X_RANGE, Y_RANGE = str(file.readlines()).strip("[]'").split('+++')
file.close()
pd.options.mode.chained_assignment = None


def plot_df(df, category_name, a):
    """
    This function takes a filtered DataFrame, on which the K-means algorithms predict labels based on the PCA
    transformation. Afterwards, the data would be plotted on a Axes object to create multiple subplots, with the
    objective of visualizing whether the created classes correspond to the target variables of interest.

    :param pd.DataFrame df: Filtered DataFrame, which contains data from an specific category.
    :param string category_name: Name of the category to use it as the scatter plot's title.
    :param matplotlib.axes.Axes a: An Axes object from a subplot grid, on which the data would be plotted.
    """

    # The PCA-transformed DataFrame and the model are used to measure each class prevalence on the given DataFrame.
    df = PCA(2).fit_transform(df)
    labels = model.predict(df)
    dict_probability = {'Probability_0': str(round((labels == 0).sum()/len(labels) * 100, 2)) + ' %',
                        'Probability_1': str(round((labels == 1).sum()/len(labels) * 100, 2)) + ' %',
                        'Probability_2': str(round((labels == 2).sum()/len(labels) * 100, 2)) + ' %',
                        'Mode': max(set(list(labels)), key=list(labels).count)}

    # Using the predicted labels, a labeled plot would be created, using the following matplotlib default colors:
    # (Class 0 : Blue), (Class 1 : Orange), (Class 2 : Green), to ensure using the same colors.
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for curr_label in np.unique(labels):
        a.scatter(df[labels == curr_label, 0], df[labels == curr_label, 1], color=colors[curr_label],
                  label='Class ' + str(curr_label) + ' (' + dict_probability['Probability_' + str(curr_label)] + ')')

    # The Axes object is used as a figure from matplotlib to dump all the gathered data and labels.
    a.set_title('Predicted classes on: {}, Mode: {}'.format(category_name, str(dict_probability['Mode'])))
    a.set_xlabel('PCA 1')
    a.set_ylabel('PCA 2')
    a.set_xlim(float(X_RANGE.split(',')[0]), float(X_RANGE.split(',')[1]))
    a.set_ylim(float(Y_RANGE.split(',')[0]), float(Y_RANGE.split(',')[1]))
    a.legend()


# The dataset is read, and the score columns are dropped, as they are not useful for the current categorization.
X = pd.read_csv('processed/combined_df.csv').drop(['MCE_Score', 'PSI_Score'], axis=1)

# For each category, a grid plot would be generated, on which their unique categories would be a separate plot on the
# grid plot. It is important to share X and Y axis, to visualize the same data on every plot and so a comparison
# could be immediately established when the subplot are being seen together.
for column in ['MCE_Category', 'PSI_Category']:
    fig = plt.figure(figsize=(10, 15))
    gs = fig.add_gridspec(len(set(X[column])), 1, wspace=0.1, hspace=0.3)
    axs = gs.subplots(sharex=True, sharey=True)
    for axis_place, category in enumerate(np.unique(X[column])):
        # The dataset is filtered to the current category and the score columns are removed, and so only source
        # variables are passed onto the function, as a PCA would be applied on continuous data.
        curr_X = X.loc[X[column] == category, :]
        curr_X = curr_X.iloc[:, :-2]
        plot_df(curr_X, category, axs[axis_place - 1])
    fig.suptitle('Grid plot for each {} category'.format(column[:3]), fontsize=20)
    fig.savefig('figures/k-means/' + column + '_k-means_predictions.png', bbox_inches="tight")
    print('Finished {} grid plot, saved on figures/k-means folder.'.format(column))
