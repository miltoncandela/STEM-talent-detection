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

categorias = {1 : 'Malo',
              2 : 'Insuficiente',
              3 : 'Regular',
              4 : 'Bueno',
              5 : 'Excelente'}

# Robotica, Programacion, Diseno 3D
psicometrico = {'Alejandro Contreras' : ['DJ04', 0, 0.6, -0.3],
          'Arturo Sanchez' : ['ST01', 0.6, 0.1, -0.1],
          'Emiliano Ruiz' : ['DJ01', 0.2, 0.7, -0.3],
          'Ernesto Daniel' : ['MT01', 1.3, 0.6, 0.4],
          'Evelyn Rosas' : ['DJ03', 0.3, 0, -0.1],
          'Israel Torres' : ['DT02', 0.1,0.8,-0.2],
          'Jezael Montano' : ['DT01', -0.3,-0.9,-0.1],
          'Joaquin Orrante' : ['DJ02', -0.5,-0.1,-0.6],
          'Jorge Ortega' : ['MJ01', 1.3,0.6,0.4],
          'Luca Rocha' : ['EJ01', np.nan, np.nan, np.nan],
          'Marcelo Contreras' : ['ES01', 0.4,0.5,0.6],
          'Mateo Rodriguez' : ['ES02', 0.5,-0.1,-0.2],
          'Patricio Sadot' : ['DT03', 0.8,-0.1,0.1],
          'Sofia Galvan' : ['MJ02', 0.5,0.5,-0.3]}
df_psciometrico = pd.DataFrame(psicometrico).T.reset_index()
df_psciometrico.columns = ['Nombre', 'Clave', 'Robot', 'Progra', 'Diseno']
print(df_psciometrico.head())

MCE = {'Alejandro Contreras' : ['DJ04', 2.5, 2.25, 2.5],
          'Arturo Sanchez' : ['ST01', 1.75, 2.5, 2.5],
          'Emiliano Ruiz' : ['DJ01', 4.25, 2.75, 4.5],
          'Ernesto Daniel' : ['MT01', 4, 3.75, 4.5],
          'Evelyn Rosas' : ['DJ03', 2.75, 3, 4.75],
          'Israel Torres' : ['DT02', 3.5, 3.75, 5],
          'Jezael Montano' : ['DT01', 3.5, 3, 4.75],
          'Joaquin Orrante' : ['DJ02', 2.25, 2.5, 3.5],
          'Jorge Ortega' : ['MJ01', 3.5, 3, 4],
          'Luca Rocha' : ['EJ01', 0,0,0],
          'Marcelo Contreras' : ['ES01', 3, 3.5, 4.5],
          'Mateo Rodriguez' : ['ES02', 3, 3, 3.75],
          'Patricio Sadot' : ['DT03', 3.75, 4, 4.25],
          'Sofia Galvan' : ['MJ02', 3.75, 3, 4]}
df_MCE = pd.DataFrame(MCE).T.reset_index()
df_MCE.columns = ['Nombre', 'Clave', 'Robot', 'Progra', 'Diseno']
print(df_MCE.head())

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
def plot_filt(dfs, nombre, num, toma, score_psicometrico, score_MCE, a):
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

    goodIDX = range(dfPPG.shape[0])
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
            df_names = df[['Nombre', 'Num', 'Toma', 'Ses']]
            df.drop(['Nombre', 'Num', 'Toma', 'Ses'], axis=1, inplace=True)
            columnas = df.columns
            escalador = load(open('esc' + ID + '.pkl', 'rb'))
            df = pd.DataFrame(escalador.transform(df))
            df.columns = columnas
            df = df.applymap(sigmoid)
            return(df, df_names)
        dfPPG, _ = df_to_prom(dfPPG, 'PPG')
        dfEEG, df_names = df_to_prom(dfEEG, 'EEG')

        dfCV['Minute'] = (dfCV.loc[:,'Second']/60).apply(math.ceil)
        dfCV['Bin'] = pd.cut(x = dfCV.loc[:,'Minute'], bins = len(dfPPG.index), labels = [x for x in range(1, len(dfPPG.index) + 1)])

        proporciones = pd.DataFrame(dfCV.groupby('Bin').EmotionDetected.apply(func = prom)).unstack().reset_index()
        del proporciones['Bin']
        proporciones = proporciones['EmotionDetected']

        dfcomb = pd.concat([dfPPG, proporciones, dfEEG, df_names], axis = 1)
        return(dfcomb)
    combdf = CombDFs(dfPPG, dfCV, dfEEG)
    combdf['ID'] = combdf.Nombre + combdf.Num
    combdf.drop(['Nombre', 'Num', 'Ses'], axis=1, inplace=True)

    if (score_psicometrico is not None) or (score_MCE is not None):
        if score_psicometrico is None:
            df_score = df_MCE
            score = score_MCE
        else:
            df_score = df_psciometrico
            score = score_psicometrico
        combdf = combdf.merge(df_score, left_on = 'ID', right_on = 'Clave', how = 'left')

        def get_score(toma, score):
            if int(toma) == 1:
                return (score[0])
            elif int(toma) == 2:
                return (score[1])
            elif int(toma) == 4:
                return (score[2])

        combdf['Score'] = [get_score(combdf.iloc[i, combdf.columns.get_loc('Toma')],
                                     combdf.iloc[i, [combdf.columns.get_loc('Robot'),
                                                     combdf.columns.get_loc('Progra'),
                                                     combdf.columns.get_loc('Diseno')]]) for i in range(combdf.shape[0])]

        combdf.drop(['Nombre', 'Clave', 'Robot', 'Progra', 'Diseno'], axis = 1, inplace = True)
        combdf.dropna(axis = 0, inplace = True)
        if score_psicometrico is None:
            combdf['Categoria'] = pd.cut(combdf.Score, [0, 1, 2, 3, 4, 5], labels = list(range(5)))
        else:
            combdf['Categoria'] = pd.cut(combdf.Score, bins = 5, labels = list(range(5)))
        print(combdf.Categoria)
        combdf.drop(['Toma', 'ID', 'Score'], axis = 1, inplace = True)
        combdf = combdf[combdf.Categoria == score]
        if combdf.shape[0] == 0:
            return(0)
        print(combdf.head())
        combdf.drop('Categoria', axis = 1, inplace = True)

    df = PCA(2).fit_transform(combdf)

    pred = modelo.predict(df)
    dict_probas = {'Probability_0' : str(round((pred == 0).sum()/len(pred) * 100, 2)) + ' %',
                   'Probability_1': str(round((pred == 1).sum()/len(pred) * 100, 2)) + ' %',
                   'Probability_2': str(round((pred == 2).sum()/len(pred) * 100, 2)) + ' %',
                   'Probability_3': str(round((pred == 3).sum()/len(pred) * 100, 2)) + ' %',
                   'Probability_4': str(round((pred == 4).sum()/len(pred) * 100, 2)) + ' %',
                   'Mode': max(set(list(pred)), key = list(pred).count)}
    print(dict_probas)

    # Class 0 : Blue
    # Class 1 : Orange
    # Class 2 : Green

    file1 = open('dimensiones.txt', 'r')
    rango_x, rango_y = str(file1.readlines()).strip("[]'").split('+++')
    file1.close()

    labels = modelo.predict(df)
    u_labels = np.unique(labels)

    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in u_labels:
        a.scatter(df[labels == i , 0] , df[labels == i , 1] , label = 'Class ' + str(i) + ' (' + dict_probas['Probability_' + str(i)] + ')', color = colores[i])
    atributos = [nombre, num, toma, score_psicometrico, score_MCE, str(dict_probas['Mode'])]
    for i, atributo in enumerate(atributos):
        if atributo is None:
            atributos[i] = 'All'
        atributos[i] = str(atributos[i])
    a.set_title('Predicted Class on: Name: {}, Num: {}, Take: {}, Psico: {}, MCE: {}, Mode: {}'.format(*atributos))
    a.set_xlabel('PCA 1')
    a.set_ylabel('PCA 2')
    a.set_xlim([float(rango_x.split(',')[0]), float(rango_x.split(',')[1])])
    a.set_ylim([float(rango_y.split(',')[0]), float(rango_y.split(',')[1])])
    a.legend()

def plot_names():
    fig = plt.figure()
    gs = fig.add_gridspec(3,2, wspace = 0.1, hspace=0.3)
    axs = gs.subplots(sharex=True, sharey=True)
    plot_filt(dfs, 'DJ', None, None, None, None, axs[0,0])
    plot_filt(dfs, 'DT', None, None, None, None, axs[1,0])
    plot_filt(dfs, 'ES', None, None, None, None, axs[2,0])
    plot_filt(dfs, 'MJ', None, None, None, None, axs[0,1])
    plot_filt(dfs, 'MT', None, None, None, None, axs[1,1])
    plot_filt(dfs, 'ST', None, None, None, None, axs[2,1])
    plt.show()

def plot_takes():
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, wspace = 0.1, hspace=0.3)
    axs = gs.subplots(sharex=True, sharey=True)
    plot_filt(dfs, None, None, 1, None, None, axs[0,0])
    plot_filt(dfs, None, None, 2, None, None, axs[0,1])
    #plot_filt(dfs, None, None, 3, None, None, axs[0,1])
    plot_filt(dfs, None, None, 4, None, None, axs[1,1])
    plt.show()

def plot_psico():
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2, wspace = 0.1, hspace=0.3)
    axs = gs.subplots(sharex=True, sharey=True)
    plot_filt(dfs, None, None, None, 0, None, axs[0,0])
    plot_filt(dfs, None, None, None, 1, None, axs[1,0])
    plot_filt(dfs, None, None, None, 2, None, axs[2,0])
    plot_filt(dfs, None, None, None, 3, None, axs[0,1])
    plot_filt(dfs, None, None, None, 4, None, axs[1,1])
    plt.show()

def plot_MCE():
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2, wspace = 0.1, hspace=0.3)
    axs = gs.subplots(sharex=True, sharey=True)
    plot_filt(dfs, None, None, None, None, 0, axs[0,0])
    plot_filt(dfs, None, None, None, None, 1, axs[1,0])
    plot_filt(dfs, None, None, None, None, 2, axs[2,0])
    plot_filt(dfs, None, None, None, None, 3, axs[0,1])
    plot_filt(dfs, None, None, None, None, 4, axs[1,1])
    plt.show()
plot_MCE()