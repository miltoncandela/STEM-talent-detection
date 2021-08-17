import pandas as pd
import matplotlib.pyplot as plt
df_results = pd.read_csv('processed/MCE_Score_Results.csv')

plt.scatter(x=df_results.index.astype('int'), y=df_results.RF_Train)
plt.scatter(x=df_results.index.astype('int'), y=df_results.RF_Test)
plt.show()