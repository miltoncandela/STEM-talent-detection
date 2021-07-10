# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import os
from sklearn.cluster import KMeans

#dfPPG = pd.read_csv('datos1/DJ01-EM01-A_ppgResults.csv', na_values =  '--')
#dfCV = pd.read_csv('datos1/DJ01-VI01-A_dlib.csv')

#pd.set_option('display.max_rows', 500)

def getDF(tipo, carpeta):
    camino = 'datos/' + tipo + '/' + carpeta + '/'
    lista_archivos = os.listdir(camino)

    #ids = [x[0:2] for x in lista_archivos]

    df = pd.read_csv(camino + lista_archivos[0], na_values =  '--')
    df['Nombre'] = np.ones(len(df.index))
    df['Num'] = np.ones(len(df.index))
    df['Ses'] = np.ones(len(df.index))

    columnas = df.columns
    #print(columnas)
    #columnas.append('Nombre')

    df = pd.DataFrame(data = np.ones(len(columnas)).T, index = columnas)
    df = df.T

    for i, archiv in enumerate(lista_archivos):
        df_temp = pd.read_csv(camino + archiv, na_values = '--')
        df_temp['Nombre'] = archiv[0:2]
        df_temp['Num'] = archiv[2:4]
        df_temp['Ses'] = archiv[10]
        if archiv[0:2] != 'no':
            df = pd.concat([df, df_temp], ignore_index = True)
    df.drop(0, axis = 0, inplace = True)
    df.dropna(axis = 1, how='any', inplace = True)
    df.index = range(len(df.index))
    df.Num = pd.to_numeric(df.Num)
    #df.index = range(0, len(df.iloc[:, 1]))
    return(df)

dfPPG1 = getDF('Empatica', 'Resultados Primera Toma')
dfPPG2 = getDF('Empatica', 'Resultados Segunda Toma')
dfPPG3 = getDF('Empatica', 'Resultados Tercera Toma')
dfPPG4 = getDF('Empatica', 'Resultados Cuarta Toma')
dfCV1 = getDF('Emociones', 'Resultados Primera Toma DLIB')
dfCV2 = getDF('Emociones', 'Resultados Segunda Toma DLIB')
dfCV3 = getDF('Emociones', 'Resultados Tercera Toma DLIB')
dfCV4 = getDF('Emociones', 'Resultados Cuarta Toma DLIB')

dfPPG = pd.concat([dfPPG1, dfPPG2, dfPPG3, dfPPG4], ignore_index = True)
dfCV = pd.concat([dfCV1, dfCV2, dfCV3, dfCV4], ignore_index = True)

dfPPG.dropna(axis = 0, how='any', inplace = True)
dfCV.dropna(axis = 0, how='any', inplace = True)

dfPPG.dropna(axis = 1, how='any', inplace = True)
dfCV.dropna(axis = 1, how='any', inplace = True)

dfPPG.index = range(len(dfPPG.index))
dfCV.index = range(len(dfCV.index))

print(dfPPG)
print(dfCV)

#print(dfPPG1)
#print(dfPPG2)
#print(dfPPG3)

def sigmoid(x):
    return(1/(1+np.exp(x)))
def prom(x):
    x = x[x != 'Pass']
    if len(x) == 0:
        return({'cNeu' : 0, 'cSur' : 0, 'cSad' : 0, 'cHap' : 0, 'cFea' : 0, 'cAng' : 0})
    prop = {'cNeu' : len(x[x == 'neutral'])/len(x), # 0 Neutral
              'cSur' : len(x[x == 'surprise'])/len(x), # 1 Surprised
              'cSad' : len(x[x == 'sad'])/len(x), # 2 Sad
              'cHap' : len(x[x == 'happy'])/len(x), # 3 Happy
              'cFea' : len(x[x == 'fear'])/len(x), # 4 Fear
              'cAng' : len(x[x == 'angry'])/len(x) # 6 Angry
    }
    return(prop)
def CombDFs(dfPPG, dfCV):
    #dfPPG.sdsd = pd.to_numeric(dfPPG.sdsd)
    #dfPPG = dfPPG.dropna(axis = 1, how = 'any').drop('Segment_Indices', axis = 1).reset_index()

    del dfPPG['Segment_Indices']
    dfPPG.drop(['Nombre', 'Num', 'Ses'], axis=1, inplace=True)
    columnas = dfPPG.columns
    dfPPG = pd.DataFrame(StandardScaler().fit_transform(dfPPG))
    dfPPG.columns = columnas
    #print(dfPPG.describe())
    dfPPG = dfPPG.applymap(sigmoid)

    dfCV['Minute'] = (dfCV.Second/60).apply(math.ceil)
    dfCV['Bin'] = pd.cut(x = dfCV.Minute, bins = len(dfPPG.index), labels = [x for x in range(1, len(dfPPG.index) + 1)])

    proporciones = pd.DataFrame(dfCV.groupby('Bin').EmotionDetected.apply(func = prom)).unstack().reset_index()
    del proporciones['Bin']
    proporciones = proporciones['EmotionDetected']

    dfcomb = pd.concat([dfPPG, proporciones], axis = 1)

    return(dfcomb)
combdf = CombDFs(dfPPG, dfCV)
print(combdf)
print(combdf.describe())

from sklearn.decomposition import PCA

pca = PCA(2)

df = pca.fit_transform(combdf)
print(df)

from sklearn.metrics import silhouette_score

k_max = 10
scores = [silhouette_score(df, KMeans(n_clusters = n).fit(df).labels_) for n in range(2,k_max)]
inertias = [KMeans(n_clusters = n).fit(df).inertia_ for n in range(2,k_max)]

import matplotlib.pyplot as plt

plt.scatter(df[:,0] , df[:,1])
plt.title('Datos procesados de $\it{Empatica}$ y $\it{VC}$ para la cuarta toma')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

plt.plot(range(2,k_max), scores, 'co-', linewidth = 2, markersize = 7)
plt.title('Silhouette score respecto a k')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.legend()
plt.show()

plt.plot(range(2,k_max), inertias, 'co-', linewidth = 2, markersize = 7)
plt.title('Inertia respecto a k')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.legend()
plt.show()

modelo = KMeans(n_clusters = 5).fit(df)
labels = modelo.predict(df)

centroids = modelo.cluster_centers_
u_labels = np.unique(labels)

for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1] , label = 'Class ' + str(i))
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.title('Processed $\it{Empatica}$ and $\it{CV}$ (Emotions)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()