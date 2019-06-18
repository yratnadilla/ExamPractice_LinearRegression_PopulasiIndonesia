import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# create first dataframe
df = pd.read_excel(
    'indo_12_1.xls', 
    skiprows=3, 
    skipfooter= 2, 
    na_values= '-'
)

df = df.rename(columns= {'Unnamed: 0' : 'Provinsi'})
nation = df[df.index == 33]     # data for Indonesia


# create second dataframe (excluding Indonesia)
df2 = df[:-1]


# filter second dataframe to search for province with highest number of population in 2010
max_2010 = df2[df2[2010] == df2[2010].max()]


# filter second dataframe to search for province with lowest number of population in 1971
min_1971 = df2[df2[1971] == df2[1971].min()]


# create dataframe for plotting
dataplot = pd.concat([max_2010, min_1971, nation])
dataplot = dataplot.set_index('Provinsi')

dataplotT = dataplot.T
dataplotT['Tahun'] = dataplotT.index


# build model with simple linear regression
# model 1 --> model to predict population of province with highest number of population in 2010
model1 = linear_model.LinearRegression()
model1.fit(dataplotT[['Tahun']], dataplotT.iloc[0:, 0])
dataplotT['Prov1_Best'] = model1.predict(dataplotT[['Tahun']])

# predict population in 2050
pred1 = model1.predict([[2050]])
pred1 = int(pred1)
print('Prediksi jumlah penduduk di', dataplotT.columns[0], 'tahun 2050 adalah', pred1)


# model 2 --> model for population of province with lowest number of population in 1971
model2 = linear_model.LinearRegression()
model2.fit(dataplotT[['Tahun']], dataplotT.iloc[0:, 1])
dataplotT['Prov2_Best'] = model2.predict(dataplotT[['Tahun']])

# predict its population in 2050
pred2 = model2.predict([[2050]])
pred2 = int(pred2)
print('Prediksi jumlah penduduk di', dataplotT.columns[1], 'tahun 2050 adalah', pred2)


# model 3 --> model for population of Indonesia
model3 = linear_model.LinearRegression()
model3.fit(dataplotT[['Tahun']], dataplotT.iloc[0:, 2])
dataplotT['Prov3_Best'] = model3.predict(dataplotT[['Tahun']])

# predict its population in 2050
pred3 = model3.predict([[2050]])
pred3 = int(pred3)
print('Prediksi jumlah penduduk di', dataplotT.columns[2], 'tahun 2050 adalah', pred3)

# plot data with trend (best-fit) line
plt.title('Populasi Penduduk di Indonesia')
plt.plot(
    dataplotT['Tahun'], dataplotT.iloc[0:, 0], 'g-o',
    dataplotT['Tahun'], dataplotT.iloc[0:, 1], 'r-o',
    dataplotT['Tahun'], dataplotT.iloc[0:, 2], 'b-o',
    dataplotT['Tahun'], dataplotT['Prov1_Best'], 'y-',
    dataplotT['Tahun'], dataplotT['Prov2_Best'], 'y-',
    dataplotT['Tahun'], dataplotT['Prov3_Best'], 'y-'
)

plt.grid(True)
plt.xlabel('Tahun')
plt.ylabel('Penduduk (dalam ratus juta jiwa)')
plt.legend([dataplotT.columns[0], dataplotT.columns[1], dataplotT.columns[2], 'Best Fit Line'])
plt.show()