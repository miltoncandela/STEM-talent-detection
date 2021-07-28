
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import os
from sklearn.cluster import KMeans

#pd.set_option('display.max_rows', 500)

def getDF(tipo, carpeta):
    camino = 'datos/' + tipo + '/' + carpeta + '/'
    lista_archivos = os.listdir(camino)

    df = pd.read_csv(camino + lista_archivos[0], na_values =  '--')
    df['Nombre'] = np.ones(len(df.index))
    df['Num'] = np.ones(len(df.index))
    df['Toma'] = np.ones(len(df.index))
    df['Ses'] = np.ones(len(df.index))

    columnas = df.columns

    df = pd.DataFrame(data = np.ones(len(columnas)).T, index = columnas)
    df = df.T

    l = []
    for i, archiv in enumerate(lista_archivos):
        df_temp = pd.read_csv(camino + archiv, na_values = '--')
        if tipo == 'EEG' or tipo == 'Empatica':
            l.append(df_temp.shape[0])
            #print(df_temp.shape)
            #df_temp = df_temp[:-5]
            #print(df_temp.shape)
            #df_temp.drop(df_temp.tail(1).index, inplace = True)
        df_temp['Nombre'] = archiv[0:2]
        df_temp['Num'] = archiv[2:4]
        df_temp['Toma'] = archiv[8]
        df_temp['Ses'] = archiv[10]
        if archiv[0:2] != 'no':
            df = pd.concat([df, df_temp], ignore_index = True)
    df.drop(0, axis = 0, inplace = True)
    df.dropna(axis = 1, how='any', inplace = True)
    df.index = range(len(df.index))
    df.Num = pd.to_numeric(df.Num)
    #df.index = range(0, len(df.iloc[:, 1]))
    return(df, l)
def getDFTomas(tipo, carpetas):
    l = []
    df, l_temp = getDF(tipo, carpetas.pop(0))
    l.append(l_temp)
    for carpeta in carpetas:
        df_temp, l_temp = getDF(tipo, carpeta)
        l.append(l_temp)
        df = pd.concat([df, df_temp], ignore_index=True)
    if tipo == 'EEG':
        df.columns = [str('EEG' + str(n)) for n in range(1, len(df.columns) - 3)] + ['Nombre', 'Num', 'Toma', 'Ses']
        goodIDX = range(dfPPG.shape[0])
        df = df.iloc[goodIDX, :]
    df = df.sort_values(by = ['Nombre', 'Num', 'Toma', 'Ses'])
    df.dropna(axis = 1, how = 'any', inplace = True)
    df.index = range(len(df.index))
    return(df, l)

dfPPG, l_PPG = getDFTomas('Empatica',['Resultados Primera Toma', 'Resultados Segunda Toma', 'Resultados Cuarta Toma'])
dfEEG, l_EEG = getDFTomas('EEG',['Toma 1', 'Toma 2', 'Toma 4'])
dfCV, l_CV = getDFTomas('Emociones',['Resultados Primera Toma DLIB', 'Resultados Segunda Toma DLIB', 'Resultados Cuarta Toma DLIB'])
print('Filas por documento en Empatica:', l_PPG)
print('Filas por documento en EEG:', l_EEG)
print('Diferencia de filas (EGG - Empatica):', [list(np.array(l_EEG[n]) - np.array(l_PPG[n])) for n in range(len(l_PPG))])

print(dfEEG.shape, dfPPG.shape, dfCV.shape)
print(dfEEG.head())
print(dfPPG.head())
print(dfCV.head())

file1 = open("columnasPPG.txt","w")
file1.writelines(','.join(dfPPG.columns))
file1.close()

file1 = open("columnasEEG.txt","w")
file1.writelines(','.join(dfEEG.columns))
file1.close()

from pickle import dump

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
def CombDFs(dfPPG, dfCV, dfEEG):
    #dfPPG.sdsd = pd.to_numeric(dfPPG.sdsd)
    #dfPPG = dfPPG.dropna(axis = 1, how = 'any').drop('Segment_Indices', axis = 1).reset_index()

    del dfPPG['Segment_Indices']

    def df_to_prom(df, ID):
        df.drop(['Nombre', 'Num', 'Toma', 'Ses'], axis=1, inplace=True)
        columnas = df.columns
        escalador = StandardScaler().fit(df)
        dump(escalador, open('esc' + ID + '.pkl', 'wb'))
        df = pd.DataFrame(escalador.transform(df))
        df.columns = columnas
        df = df.applymap(sigmoid)
        return(df)
    dfPPG = df_to_prom(dfPPG, 'PPG')
    dfEEG = df_to_prom(dfEEG, 'EEG')

    dfCV['Minute'] = (dfCV.Second/60).apply(math.ceil)
    dfCV['Bin'] = pd.cut(x = dfCV.Minute, bins = len(dfPPG.index), labels = [x for x in range(1, len(dfPPG.index) + 1)])

    proporciones = pd.DataFrame(dfCV.groupby('Bin').EmotionDetected.apply(func = prom)).unstack().reset_index()
    del proporciones['Bin']
    proporciones = proporciones['EmotionDetected']

    dfcomb = pd.concat([dfPPG, dfEEG, proporciones], axis = 1)
    return(dfcomb)

    #return(dfPPG, dfEEG, proporciones)
combdf = CombDFs(dfPPG, dfCV, dfEEG)
#dfPPG, dfEEG, dfCV = CombDFs(dfPPG, dfCV, dfEEG)

print(combdf.columns)
print(combdf.shape[1])
print(combdf.head())

import seaborn as sns
import matplotlib.pyplot as plt
correlation_mat = combdf.corr()

sns.heatmap(correlation_mat, annot = False)
plt.title('Correlation matrix of features (scaled)')
plt.show()

from sklearn.decomposition import PCA

#combdf = pd.DataFrame(columns = ['PPG', 'EEG', 'CV'], index = range(dfPPG.shape[0]))
#combdf['PPG'] = PCA(1).fit_transform(dfPPG)
#combdf['EEG'] = PCA(1).fit_transform(dfEEG)
#combdf['CV'] = PCA(1).fit_transform(dfCV)

#print(combdf.head())
pca = PCA(2)
df = pca.fit_transform(combdf)
#print(df[:5,:])

from sklearn.metrics import silhouette_score

k_max = 10
scores = [silhouette_score(df, KMeans(n_clusters = n).fit(df).labels_) for n in range(2,k_max)]
inertias = [KMeans(n_clusters = n).fit(df).inertia_ for n in range(2,k_max)]

plt.scatter(df[:,0] , df[:,1])
plt.title('Datos procesados de $\it{Empatica}$, $\it{EGG}$ y $\it{VC}$ para la 1,2,4 toma')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
#ax = plt.gca()
#x_range = ax.get_xlim()
#y_range = ax.get_ylim()
#print(x_range, y_range)
plt.show()

#file1 = open("dimensiones.txt","w")
#file1.writelines(str(x_range[0]) + ',' + str(x_range[1]) + '+++' + str(y_range[0]) + ',' + str(y_range[1]))
#file1.close()

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

k = np.argmax(scores) + 2
modelo = KMeans(n_clusters = k).fit(df)
labels = modelo.predict(df)

centroids = modelo.cluster_centers_
u_labels = np.unique(labels)

for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1] , label = 'Class ' + str(i))
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k', label = 'Centroid')
plt.title('Processed $\it{Empatica}$, $\it{EEG}$ and $\it{CV}$ (Emotions)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

dump(modelo, open('modeloKmedias.pkl', 'wb'))
