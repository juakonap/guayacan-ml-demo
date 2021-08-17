import datetime
import pandas as pd
from numpy import random
import random


start = datetime.datetime.strptime("01-01-2020", "%d-%m-%Y")
end = datetime.datetime.strptime("31-08-2021", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_format = []


for date in date_generated:
    date_format.append(date.strftime("%d-%m-%Y"))


#Serie de datos simulada
minimos = [150,10,100,10, 250, 10]


base1 = random.sample(range(minimos[0], minimos[0] + len(date_format)), len(date_format))
serie1 = [random.random() + float(x) for x in base1]

base2 = random.sample(range(minimos[1], minimos[1] + len(date_format)), len(date_format))
serie2 = [random.random() + float(x) for x in base2]

base3 = random.sample(range(minimos[2], minimos[2] + len(date_format)), len(date_format))
serie3 = [random.random() + float(x) for x in base3]

base4 = random.sample(range(minimos[3], minimos[3] + len(date_format)), len(date_format))
serie4 = [random.random() + float(x) for x in base4]



df1 = pd.DataFrame(date_format)
df1['serie1'] = pd.Series(serie1).rolling(14).mean()
df1['serie2'] = pd.Series(serie2).rolling(14).mean()
df1['serie3'] = pd.Series(serie3).rolling(14).mean()
df1['serie4'] = pd.Series(serie4).rolling(14).mean()

df1 = df1\
    .dropna(how = 'any')\
        .rename(columns={0:'date'})
    

df1.to_csv('./data/series.csv', sep = ',', decimal = '.', index = False)


#Serie de datos para predicción
df2 = pd.DataFrame(date_format)
df2['original'] = pd.Series(random.sample(range(minimos[4], minimos[4] + len(date_format)), len(date_format)-300) + random.sample(range(minimos[5], minimos[5] + len(date_format)), len(date_format) - 308)).rolling(7).mean()
df2['predict'] = pd.Series([random.random()*100 + float(x) for x in df2['original']]).rolling(7).mean()

df2 = df2\
    .dropna(how = 'any')\
        .rename(columns={0:'date'})

df2.to_csv('./data/predict.csv', sep = ';', decimal = ',', index = False)


#Serie de datos para calcular correlacion
n_data = 100
c1 = list(random.exponential(scale=1.0, size=n_data))
c2 = list(random.exponential(scale=2.0, size=n_data))
c3 = list(random.normal(loc=2.0, scale=1.5, size=n_data))
c4 = list(random.normal(loc=10.0, scale=5.0, size=n_data))
c5 = list(random.normal(loc=15.0, scale=2.5, size=n_data))

df3 = pd.DataFrame([c1,c2,c3,c4,c5]).T\
    .rename(columns={0:'Sensor 01',1:'Sensor 02',2:'Sensor 03',3:'Sensor 04',4:'Sensor 05'})

df3.corr().to_csv('./data/corr.csv', sep = ',', decimal = '.', index = True)

