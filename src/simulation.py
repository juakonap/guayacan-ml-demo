import datetime
import pandas as pd
from numpy import random
import random


start = datetime.datetime.strptime("01-01-2021", "%m-%d-%Y")
end = datetime.datetime.strptime("07-31-2021", "%m-%d-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_format = []


for date in date_generated:
    date_format.append(date.strftime("%m-%d-%Y"))


#Serie de datos simulada

base1 = random.sample(range(100, 350), len(date_format))
serie1 = [random.random() + float(x) for x in base1]

base2 = random.sample(range(10, 250), len(date_format))
serie2 = [random.random() + float(x) for x in base2]

base3 = random.sample(range(100, 350), len(date_format))
serie3 = [random.random() + float(x) for x in base3]

base4 = random.sample(range(10, 400), len(date_format))
serie4 = [random.random() + float(x) for x in base4]



df1 = pd.DataFrame(date_format)
df1['serie1'] = serie1
df1['serie2'] = serie2
df1['serie3'] = serie3
df1['serie4'] = serie4

df1 = df1\
    .rename(columns={0:'date'})
    
df1.to_csv('./data/data01.csv', sep = ',', decimal = '.', index = False)


#Serie de datos para predicción
df2 = pd.DataFrame(date_format)
df2['original'] = random.sample(range(250, 400), 100) + random.sample(range(10, 200), len(date_format) - 100)
df2['predict'] = [random.random()*100 + float(x) for x in df2['original']]

df2 = df2\
    .rename(columns={0:'date'})

df2.to_csv('./data/predict.csv', sep = ',', decimal = '.', index = False)


#Serie de datos para calcular correlacion
n_data = 100
c1 = list(random.exponential(scale=1.0, size=n_data))
c2 = list(random.exponential(scale=2.0, size=n_data))
c3 = list(random.normal(loc=2.0, scale=1.5, size=n_data))
c4 = list(random.normal(loc=10.0, scale=5.0, size=n_data))
c5 = list(random.normal(loc=15.0, scale=2.5, size=n_data))

df3 = pd.DataFrame([c1,c2,c3,c4,c5]).T\
    .rename(columns={0:'Sensor 01',1:'Sensor 02',2:'Sensor 03',3:'Sensor 04',4:'Sensor 05'})

import seaborn as sns
sns.heatmap(df3.corr(),vmin=-1, vmax=1, annot=True)

df3.to_csv('./data/corr.csv', sep = ',', decimal = '.', index = False)

