# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code uses a processed CSV file from the "processed" folder, created using the "createCSV.py" file,
# which joins the biometric devices and the predicted scores. Only source variables would be used, in order to
# create a K-means model which unifies all the biometric data into a single class.

# The processed DataFrame is read, and so target variables would be dropped, as a new class would be made.
import pandas as pd
import numpy as np

X = pd.read_csv('processed/combined_df.csv').drop(['MCE_Category', 'PSI_Category', 'PSI_Score', 'MCE_Score'], axis=1)

# A correlation matrix is created to perceive whether classes could be extracted using the current features.
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(X.corr(), annot=False)
plt.title('Correlation matrix of features (scaled)')
plt.show()

# Using Principal Component Analysis (PCA), the number of features would be reduced to 2, to plot them on a XY axis.
from sklearn.decomposition import PCA
df = PCA(2).fit_transform(X)

# Fitting multiple K-means algorithm, list comprehension is used to compute both inertia and silhouette scores,
# this scores would determine the ideal number of classes that the algorithm should have given the data.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
k_max = 10
silhouette_scores = [silhouette_score(df, KMeans(n_clusters=n).fit(df).labels_) for n in range(2, k_max)]
inertia_scores = [KMeans(n_clusters=n).fit(df).inertia_ for n in range(2, k_max)]

# Unlabeled plot would be done to show the difference between unlabeled and labeled data.
plt.scatter(df[:, 0], df[:, 1])
plt.title('Processed data from $\it{Empatica}$, $\it{EGG}$ and $\it{VC}$, takes: (1, 2, 4)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
ax = plt.gca()
x_range = ax.get_xlim()
y_range = ax.get_ylim()
plt.show()

# Plot dimension would be saved to keep the same ranges from both XY axis.
file = open("processed/PCA_dimensions.txt", "w")
file.writelines(str(x_range[0]) + ',' + str(x_range[1]) + '+++' + str(y_range[0]) + ',' + str(y_range[1]))
file.close()

# Silhouette visual representation, which would be helpful to determine the optimal number of classes.
plt.plot(range(2, k_max), silhouette_scores, 'co-', linewidth=2, markersize=7)
plt.title('Silhouette score with respect to k')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.legend()
plt.show()

# Similar to error, inertia is the error-default measure, although it is not as useful as the silhouette score.
plt.plot(range(2, k_max), inertia_scores, 'co-', linewidth=2, markersize=7)
plt.title('Inertia score with respect to k')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.legend()
plt.show()

# The best number of classes is determined by the silhouette's maximum score, from which model is trained.
k = np.argmax(silhouette_scores) + 2
model = KMeans(n_clusters=k).fit(df)

# Centroids and labels are gathered to create a labeled plot based on colors and the identified centroids.
labels = model.predict(df)
centroids = model.cluster_centers_

# The following plot combines all the previously described data into a scatter plot of labeled classes.
for curr_label in np.unique(labels):
    plt.scatter(df[labels == curr_label, 0], df[labels == curr_label, 1], label='Class ' + str(curr_label))
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k', label='Centroid')
plt.title('Processed $\it{Empatica}$, $\it{EEG}$ and $\it{CV}$ (Emotions)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# The model is saved as a pickle file with .pkl as extension, would be useful to predict on unlabeled data.
from pickle import dump
dump(model, open('processed/k-means_model.pkl', 'wb'))
