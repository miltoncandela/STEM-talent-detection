from pickle import load
modelo = load(open('modeloKmedias.pkl', 'rb'))

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pickle import load
from sklearn.decomposition import PCA
import math

pd.options.mode.chained_assignment = None

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
    df.Toma = pd.to_numeric(df.Toma)
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
    df = df.sort_values(by = ['Nombre', 'Num', 'Toma', 'Ses'])
    df.dropna(axis = 1, how = 'any', inplace = True)
    df.index = range(len(df.index))
    return(df, l)

dfPPG, l_PPG = getDFTomas('Empatica',['Resultados Primera Toma', 'Resultados Segunda Toma', 'Resultados Cuarta Toma'])
dfEEG, l_EEG = getDFTomas('EEG',['Toma 1', 'Toma 2', 'Toma 4'])
dfCV, l_CV = getDFTomas('Emociones',['Resultados Primera Toma DLIB', 'Resultados Segunda Toma DLIB', 'Resultados Cuarta Toma DLIB'])
dfs = [dfCV, dfEEG, dfPPG]
def plot_filt(dfs, nombre, num, toma, a):
    def filtDF(df, nombre, num, toma):
        valores = [nombre, num, toma]
        col_name = ['Nombre', 'Num', 'Toma']
        for i, val in enumerate(valores):
            if val is None:
                continue
            else:
                df = df[df[col_name[i]] == val]
        df.index = range(df.shape[0])
        return(df)
    dfCV, dfEEG, dfPPG = map(filtDF, dfs, [nombre] * 3, [num] * 3, [toma] * 3)

    np.random.seed(0)
    goodIDX = np.random.choice(dfEEG.index, size = dfPPG.shape[0], replace = False)
    dfEEG = dfEEG.iloc[goodIDX, :]

    file1 = open('columnasEEG.txt', 'r')
    columnasEEG = str(file1.readlines()).strip("[]'").split(',')
    file1.close()

    file1 = open('columnasPPG.txt', 'r')
    columnasPPG = str(file1.readlines()).strip("[]'").split(',')
    file1.close()

    dfEEG = dfEEG[columnasEEG]
    dfPPG = dfPPG[columnasPPG]

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
        del dfPPG['Segment_Indices']
        def df_to_prom(df, ID):
            df.drop(['Nombre', 'Num', 'Toma', 'Ses'], axis=1, inplace=True)
            columnas = df.columns
            escalador = load(open('esc' + ID + '.pkl', 'rb'))
            df = pd.DataFrame(escalador.transform(df))
            df.columns = columnas
            df = df.applymap(sigmoid)
            return(df)
        dfPPG = df_to_prom(dfPPG, 'PPG')
        dfEEG = df_to_prom(dfEEG, 'EEG')

        dfCV['Minute'] = (dfCV.loc[:,'Second']/60).apply(math.ceil)
        dfCV['Bin'] = pd.cut(x = dfCV.loc[:,'Minute'], bins = len(dfPPG.index), labels = [x for x in range(1, len(dfPPG.index) + 1)])

        proporciones = pd.DataFrame(dfCV.groupby('Bin').EmotionDetected.apply(func = prom)).unstack().reset_index()
        del proporciones['Bin']
        proporciones = proporciones['EmotionDetected']

        dfcomb = pd.concat([dfPPG, proporciones, dfEEG], axis = 1)
        return(dfcomb)
    combdf = CombDFs(dfPPG, dfCV, dfEEG)

    df = PCA(2).fit_transform(combdf)

    pred = modelo.predict(df)
    dict_probas = {'Probability_0' : str(round((pred == 0).sum()/len(pred) * 100, 2)) + ' %',
                   'Probability_1': str(round((pred == 1).sum()/len(pred) * 100, 2)) + ' %',
                   'Probability_2': str(round((pred == 2).sum()/len(pred) * 100, 2)) + ' %',
                   'Mode': max(set(list(pred)), key = list(pred).count)}
    print(dict_probas)
    # Class 0 : Blue
    # Class 1 : Orange
    # Class 2 : Green

    file1 = open('dimensiones.txt', 'r')
    rango_x, rango_y = str(file1.readlines()).strip("[]'").split('+++')
    file1.close()

    labels = modelo.predict(df)
    centroids = modelo.cluster_centers_
    u_labels = np.unique(labels)

    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in u_labels:
        a.scatter(df[labels == i , 0] , df[labels == i , 1] , label = 'Class ' + str(i) + ' (' + dict_probas['Probability_' + str(i)] + ')', color = colores[i])
    atributos = [nombre, num, toma, str(dict_probas['Mode'])]
    for i, atributo in enumerate(atributos):
        if atributo is None:
            atributos[i] = 'All'
        atributos[i] = str(atributos[i])
    a.set_title('Predicted Class on: ' + 'Name: ' + atributos[0] + ', ' + 'Num: ' + atributos[1] + ', ' + 'Take: ' + atributos[2] + ', ' + 'Mode: ' + atributos[3])
    a.set_xlabel('PCA 1')
    a.set_ylabel('PCA 2')
    #ax = plt.gca()
    a.set_xlim([float(rango_x.split(',')[0]), float(rango_x.split(',')[1])])
    a.set_ylim([float(rango_y.split(',')[0]), float(rango_y.split(',')[1])])
    a.legend()
    #plt.show()

def plot_names():
    fig = plt.figure()
    gs = fig.add_gridspec(3,2, wspace = 0.1, hspace=0.3)
    axs = gs.subplots(sharex=True, sharey=True)
    plot_filt(dfs, 'DJ', None, None, axs[0,0])
    plot_filt(dfs, 'DT', None, None, axs[1,0])
    plot_filt(dfs, 'ES', None, None, axs[2,0])
    plot_filt(dfs, 'MJ', None, None, axs[0,1])
    plot_filt(dfs, 'MT', None, None, axs[1,1])
    plot_filt(dfs, 'ST', None, None, axs[2,1])
    plt.show()

def plot_takes():
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, wspace = 0.1, hspace=0.3)
    axs = gs.subplots(sharex=True, sharey=True)
    plot_filt(dfs, None, None, 1, axs[0,0])
    plot_filt(dfs, None, None, 2, axs[0,1])
    #plot_filt(dfs, None, None, 3, axs[0,1])
    plot_filt(dfs, None, None, 4, axs[1,1])
    plt.show()
plot_takes()