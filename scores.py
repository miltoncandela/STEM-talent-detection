import pandas as pd
import numpy as np
import os
import math
from sklearn.preprocessing import StandardScaler

#pd.set_option('display.max_rows', 500)
categorias = {1 : 'Malo',
              2 : 'Insuficiente',
              3 : 'Regular',
              4 : 'Bueno',
              5 : 'Excelente'}

# Robotica, Programacion, Diseno 3D
scores = {'Alejandro Contreras' : ['DJ04', 2.5, 2.25, 2.5],
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

df_scores = pd.DataFrame(scores).T.reset_index()
df_scores.columns = ['Nombre', 'Clave', 'Robot', 'Progra', 'Diseno']
print(df_scores)

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
    #df.Num = pd.to_numeric(df.Num)
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
print(dfPPG)
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
        escalador = StandardScaler().fit(df)
        df = pd.DataFrame(escalador.transform(df))
        df.columns = columnas
        df = df.applymap(sigmoid)
        return(df, df_names)
    dfPPG, _ = df_to_prom(dfPPG, 'PPG')
    dfEEG, df_names = df_to_prom(dfEEG, 'EEG')

    dfCV['Minute'] = (dfCV.Second/60).apply(math.ceil)
    dfCV['Bin'] = pd.cut(x = dfCV.Minute, bins = len(dfPPG.index), labels = [x for x in range(1, len(dfPPG.index) + 1)])

    proporciones = pd.DataFrame(dfCV.groupby('Bin').EmotionDetected.apply(func = prom)).unstack().reset_index()
    del proporciones['Bin']
    proporciones = proporciones['EmotionDetected']

    dfcomb = pd.concat([dfPPG, dfEEG, proporciones, df_names], axis = 1)
    return(dfcomb)
combdf = CombDFs(dfPPG, dfCV, dfEEG)
combdf['ID'] = combdf.Nombre + combdf.Num
combdf.drop(['Nombre', 'Num', 'Ses'], axis = 1, inplace = True)
combdf = combdf.merge(df_scores, left_on = 'ID', right_on = 'Clave', how = 'left')

def score(toma, score):
    if int(toma) == 1:
        return(score[0])
    elif int(toma) == 2:
        return(score[1])
    elif int(toma) == 4:
        return(score[2])
combdf['Score'] = [score(combdf.iloc[i, combdf.columns.get_loc('Toma')],
                         combdf.iloc[i, [combdf.columns.get_loc('Robot'),
                                         combdf.columns.get_loc('Progra'),
                                         combdf.columns.get_loc('Diseno')]]) for i in range(combdf.shape[0])]

combdf.drop(['Nombre', 'Clave', 'Robot', 'Progra', 'Diseno'], axis = 1, inplace = True)
combdf['Categoria'] = pd.cut(combdf.Score, [0, 1, 2, 3, 4, 5], labels = categorias.values())
#combdf.drop('Score', axis = 1, inplace = True)
combdf.Toma = combdf.Toma.astype('category')
combdf.drop(['Toma', 'ID'], axis = 1, inplace = True)
combdf.dropna(axis = 0, inplace = True)
print(combdf)
combdf.to_csv('ResultadosMCE.csv', index = False)
