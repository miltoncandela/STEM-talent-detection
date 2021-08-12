# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


X = pd.read_csv('processed/combined_df.csv').drop(['MCE_Category', 'PSI_Category'], axis=1)
Y = X.loc[:, ['MCE_Score', 'PSI_Score']]
X.drop(['MCE_Score', 'PSI_Score'], axis=1, inplace=True)

print(X.head())
print(Y)

# Divides the dataset into training and testing dataset by P ratio, then it scales it using Z-Score (StandardScaler)
# it is worth noting that StandardScaler outputs an array, and so it is transformed back to a pandas DataFrame.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

P = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=P, random_state=60)
escalator = StandardScaler().fit(X_train)
df_columns = X_train.columns

X_train = pd.DataFrame(escalator.transform(X_train), columns=df_columns)
X_test = pd.DataFrame(escalator.transform(X_test), columns=df_columns)

BATCH_SIZE = 50

# Imports tensorflow library, which has deep learning function to build and train a Recurrent Neural Network, further
# code also sets up a GPU with 2GB as a virtual device for faster training, in case the user has one physical GPU.
import tensorflow as tf
available_devices = tf.config.experimental.list_physical_devices('GPU')
if len(available_devices) > 0:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_virtual_device_configuration(gpu, [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        # tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN


def create_model(name=None):
    """
    Using tensorflow and keras, this function builds a sequential model with a RNN architecture, based on layers such
    as SimpleRNN, Dropout and a final Dense layer for the output. When it finishes training, the model is saved on the
    "saved_models" folder when the name parameter is different than None, the function also plots the metrics with
    respect to the number of epochs involve during the computation.

    :param string name: Name of the file on which the model would be saved.
    :return tf.keras.models.Sequential: An already trained RNN, trained using the designated train_generator.
    """

    ann = Sequential()
    ann.add(Dense(200, activation='relu', input_shape=[len(X_train.columns)]))
    ann.add(Dropout(.3))
    ann.add(Dense(300, activation='relu'))
    ann.add(Dropout(.6))
    ann.add(Dense(300, activation='relu'))
    ann.add(Dropout(.6))
    ann.add(Dense(200, activation='relu'))
    ann.add(Dropout(.3))
    ann.add(Dense(100, activation='relu'))
    ann.add(Dropout(.15))
    ann.add(Dense(2))

    ann.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=METRICS.keys(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    history = ann.fit(X_train, y_train, validation_split=0.2,
                      epochs=EPOCH, verbose=2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    for metric in METRICS.keys():
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel(METRICS[metric])
        plt.plot(hist['epoch'], hist[metric], label='Training')
        plt.plot(hist['epoch'], hist['val_' + metric], label='Validation')
        plt.legend()
        plt.show()

    return ann


# The following chunks of code represents two ways a RNN model could be generated, either by CREATING or IMPORTING,
# please comment or uncomment the lines of code depending on the desired outcome.

METRICS = {'mae': 'Mean Absolute Error (MAE)', 'mse': 'Mean Squared Error (MSE)'}
EPOCH = 500

# CREATING: Model generation via create_model, name is a parameter to save the model on "saved_models" folder.
model = create_model()
model.summary()


def model_evaluation(predictions, true_values):
    """
    Manual evaluation of the model's predictions using R squared, pearson correlation and p-value. This is done by each
    dimension (X, Y, Z), and the function is called by each dataset (training, validation, testing).

    :param np.array predictions: Predicted forces using markers DataFrame and the trained RNN.
    :param np.array true_values: Original array of n rows by three columns (X, Y, Z) which contains forces.
    :return (list, float): A list of the three R squared scores depending on each dimension (X, Y, Z), and a rounded
    float which is the mean value of all coefficients of determination.
    """

    # predictions = np.array(pd.DataFrame({'MCE_Score': predictions[:, 0], 'PSI_Score': predictions[:, 1]}

    from sklearn.metrics import r2_score
    scores_r2 = [np.abs(r2_score(true_values[:, num], predictions[:, num])) for num in range(predictions.shape[1])]

    from scipy.stats.stats import pearsonr
    scores_pearson = [np.abs(pearsonr(true_values[:, num], predictions[:, num])[0]) for num in range(predictions.shape[1])]
    p_pearson = [np.abs(pearsonr(true_values[:, num], predictions[:, num])[1]) for num in range(predictions.shape[1])]

    print('Pearson correlation:', scores_pearson)
    print('Pearson correlation (mean):', np.round(np.mean(scores_pearson), 4))
    print('P-value:', p_pearson)
    print('P-value (mean):', np.round(np.mean(p_pearson), 4))
    print('Coefficient of determination:', scores_r2)
    print('Coefficient of determination (mean):', np.round(np.mean(scores_r2), 4))
    return scores_r2, float(np.round(np.mean(scores_r2), 4))


print('*** Training evaluation ***')
train_scores, train_det = model_evaluation(model.predict(X_train), np.array(y_train))
print('*** Testing evaluation ***')
test_scores, test_det = model_evaluation(model.predict(X_test), np.array(y_test))
det_scores = [train_det, test_det]
x_scores = [train_scores[0], test_scores[0]]
y_scores = [train_scores[1], test_scores[1]]

# The previously obtained scores, although they are printed on the console using the declared function model_evaluation,
# would be plotted using a bar plot. Each bar represents a force dimension and the set of bars represent each type of
# dataset available, and so a for loop that changes the bars position have to be employed to visually see the metrics.

bars = ('Training', 'Testing')
BAR_WIDTH = 0.15
y_pos = np.arange(len(bars))
names = ['MCE_Score', 'PSI_Score', 'Mean']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, score in enumerate([x_scores, y_scores, det_scores]):
    plt.bar([x + BAR_WIDTH*i for x in y_pos], score, width=BAR_WIDTH, label=names[i], color=colors[i])
plt.title('Coefficient of determination (R Squared) across datasets ({} Epochs)'.format(EPOCH))
plt.ylabel('Coefficient of determination ($R^{2}$)')
plt.xticks([r + BAR_WIDTH*1.5 for r in y_pos], bars)
ax = plt.gca()
ax.set_ylim([0, 1])
plt.legend()
plt.show()

# Now automatic evaluation of the model takes place, the set of metrics used correspond to the ones that were tracked
# of the training and validation dataset, collected within the training of the RNN.

y_prediction = model.predict(X_test)
y_true = np.array(y_test)
y_both = pd.DataFrame(np.hstack([y_prediction, y_true]), columns=['MCE_Prediction', 'PSI_Prediction', 'MCE_True', 'PSI_True'])
print(y_both)

test_metrics = model.evaluate(X_test, verbose=0)
print('*** Test metrics ***')
for i, test_metric in enumerate(['loss'] + list(METRICS.keys())):
    print(test_metric, test_metrics[i])


def plot_results(predictions, true_values):
    """
    Uses matplotlib to visually see whether a correlation is being observed, although a subset of the first 100 rows
    is used. As there are thousands of samples and it is nearly impossible to correctly visualize them on one plot.

    :param np.array predictions: Non-scaled predictions from the RNN.
    :param np.array true_values: Non-scaled real data from testing dataset.
    """

    for idx, score in enumerate(['MCE', 'PSI']):
        plt.plot(true_values[:, idx], 'y', label='Predicted')
        plt.plot(predictions[:, idx], 'r', label='True')
        plt.title('True and predicted {} score'.format(score))
        plt.xlabel('Index')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

        plt.plot(true_values[:, idx] - predictions[:, idx], 'b', label='Residuals')
        plt.title('Residual plot of {} score'.format(score))
        plt.xlabel('Index')
        plt.ylabel('True value - Prediction')
        plt.legend()
        plt.show()


plot_results(y_prediction, y_true)