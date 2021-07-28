import pandas as pd

categorias = {1 : 'Malo',
              2 : 'Insuficiente',
              3 : 'Regular',
              4 : 'Bueno',
              5 : 'Excelente'}

# Robotica, Programacion, Diseno 3D
scores = {'Alejandro Contreras' : ['ST01', 2.5, 2.25, 2.5],
          'Arturo Sanchez' : ['ST01', 1.75, 2.5, 2.5],
          'Emiliano Ruiz' : ['ST02', 4.25, 2.75, 4.5],
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

df = pd.DataFrame(scores).T.reset_index()
df.columns = ['Nombre', 'Clave', 'Robot', 'Progra', 'Diseno']
print(df)
