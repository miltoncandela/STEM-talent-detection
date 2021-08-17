
import pandas as pd
MCE_categories = {2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
inv_categories = {v: k for k, v in MCE_categories.items()}

channels = ['FP1', 'F3', 'C3', 'PZ', 'C4', 'F4', 'FP2']
signals = ['A', 'LB', 'HB', 'G', 'T']

# Target variables that are not currently used, would be dropped.
x = pd.read_csv('processed/combined_df.csv')
y = x.pop('MCE_Category')
x = x.iloc[:, :-3]
y = y.astype('category')

from imblearn.over_sampling import RandomOverSampler
x_resampled, y_resampled = RandomOverSampler(random_state=50).fit_resample(x, y)

from sklearn.ensemble import RandomForestClassifier
# Feature selection #
model = RandomForestClassifier(random_state=50).fit(x_resampled, y_resampled)

# This RF model computes the GINI importance index, which would be used as a feature selection method, to n features
s = pd.Series(index=x.columns, data=model.feature_importances_).sort_values(ascending=False)

# Visualization #
import seaborn as sns
import matplotlib.pyplot as plt
x_resampled = x_resampled.loc[:, s.index[:5]]

x_resampled['Categoria'] = pd.Categorical(y_resampled, categories=list(reversed(MCE_categories.values())))
# x_resampled = x_resampled.sort_values(by='Categoria', key=pd.Categorical, ascending=False)

# sns.pairplot(x_resampled, hue="Categoria", markers=['s', 's', 'o', 'o'][:len(set(y))], palette=sns.color_palette("Spectral_r", 4))
# plt.show()

x_resampled = x_resampled.reset_index(drop=True)

print(dict(zip(MCE_categories.values(), ['s', 's', 'o', 'o'])))
print(x_resampled.head())
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
            if feature_i == 'Load':
                ax.set_xlim(0, 0.665)
            if feature_j == 'Load':
                ax.set_ylim(0, 0.665)
            sns.scatterplot(x=feature_i, y=feature_j, data=x_resampled, alpha=0.5, hue='Categoria',
                            palette=sns.color_palette('Spectral_r', 4))
            # plt.scatter(x=x_resampled[feature_i], y=x_resampled[feature_j], alpha=0.5,
            #             c=x_resampled['Categoria'].map(dict(zip(list(MCE_categories.values()),
            #                                                     sns.color_palette('Spectral', 4)))))
            plt.xlabel(feature_i)
            plt.ylabel(feature_j)
            plt.title('{} and {} on MCE category'.format(feature_i, feature_j))
            plt.legend()
            plt.show()