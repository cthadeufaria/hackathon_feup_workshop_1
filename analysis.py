import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

river_level_data = pd.read_csv("workshop_1/Pavia-Nivel_do_rio_por_hora.csv")
rain_data = pd.read_csv("workshop_1/Pavia-Quantidade_chuva_por_hora.csv")

full_data = pd.merge(river_level_data, rain_data, on=['Data/Hora'])
print(full_data.head())

inundation = full_data[full_data['Precipitação atual (mm)'] != 0.]
print(inundation.head())

sns.lineplot(inundation['Precipitação atual (mm)'])
sns.lineplot(inundation['Precipitação acumulada (mm)'])
plt.show()

sns.lineplot(full_data['Precipitação atual (mm)'])
sns.lineplot(full_data['Precipitação acumulada (mm)'])
plt.show()