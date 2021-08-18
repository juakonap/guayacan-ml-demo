import datetime
import pandas as pd
from numpy import random as ran
import random
from operator import add


start = datetime.datetime.strptime("01-01-2015", "%d-%m-%Y")
end = datetime.datetime.strptime("31-08-2021", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range((end-start).days + 1) if (start + datetime.timedelta(days=x)).weekday() == 6]
#date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_format = []

for date in date_generated:
    date_format.append(date.strftime("%Y-%m-%d"))

#nombre_equipo = 'equipo_03'

#Serie de datos simulada
n = len(date_format)
ponderador = .25
minimos = [int(i*ponderador) for i in [150,10,100,10, 250, 10]]
maximos = [n+m for m in minimos]


base1 = random.sample(range(minimos[0], maximos[0]), n)
serie1 = [random.random() + float(x) for x in base1]

base2 = random.sample(range(minimos[1], maximos[1]), n)
serie2 = [random.random() + float(x) for x in base2]

base3 = random.sample(range(minimos[2], maximos[2]), n)
serie3 = [random.random() + float(x) for x in base3]

base4 = random.sample(range(minimos[3], maximos[3]), n)
serie4 = [random.random() + float(x) for x in base4]



df1 = pd.DataFrame(date_format)
df1['serie1'] = pd.Series(serie1).rolling(14).mean()
df1['serie2'] = pd.Series(serie2).rolling(14).mean()
df1['serie3'] = pd.Series(serie3).rolling(14).mean()
df1['serie4'] = pd.Series(serie4).rolling(14).mean()

df1 = df1\
    .dropna(how = 'any')\
        .rename(columns={0:'date'})
    

#Serie de datos para predicción
df2 = pd.DataFrame(date_format)
df2['original'] = pd.Series(random.sample(range(minimos[4], maximos[4]), int(n*.5)) + random.sample(range(minimos[5], maximos[5]), int(n*.5))).rolling(7).mean()
df2['predict'] = pd.Series([random.random()*100 + float(x) for x in df2['original']]).rolling(7).mean()

df2 = df2\
    .dropna(how = 'any')\
        .rename(columns={0:'date'})



#Serie de datos para calcular correlacion
n_data = 100
c1 = list(ran.exponential(scale=1.0, size=n_data))
c2 = list(ran.exponential(scale=2.0, size=n_data))
c3 = list( map(add, list(ran.normal(loc=2.0, scale=1.5, size=n_data)), list([2+5*x for x in range(n_data)])) )
c4 = list( map(add, list(ran.normal(loc=2.0, scale=1.5, size=n_data)), list([12+2*x for x in range(n_data)])) )
c5 = list(ran.normal(loc=15.0, scale=2.5, size=n_data))

df3 = pd.DataFrame([c1,c2,c3,c4,c5]).T\
    .rename(columns={0:'Sensor 01',1:'Sensor 02',2:'Sensor 03',3:'Sensor 04',4:'Sensor 05'})


#Serie de datos apilada
stk = random.sample(range(200, 200 + n), n)
stacked = [random.random() + float(x) for x in stk]


df4 = pd.DataFrame(date_format)
df4['stacked1'] = pd.Series(stacked).rolling(14).mean()
df4['stacked2'] = pd.Series([x*(1+random.random()) for x in df4['stacked1']]).rolling(14).mean()
df4['stacked3'] =pd.Series([x*(1-random.random()) for x in df4['stacked1']]).rolling(14).mean()


df4 = df4\
    .dropna(how = 'any')\
        .rename(columns={0:'date'})
    
'''
df1.to_csv(f'./data/{nombre_equipo}_series.csv', sep = ',', decimal = '.', index = False)
df2.to_csv(f'./data/{nombre_equipo}_predict.csv', sep = ',', decimal = '.', index = False)
df3.corr().to_csv(f'./data/{nombre_equipo}_corr.csv', sep = ',', decimal = '.', index = True)
df4.to_csv(f'./data/{nombre_equipo}_stacked.csv', sep = ',', decimal = '.', index = False)
'''
df1.to_csv('./data/series_f2.csv', sep = ',', decimal = '.', index = False)
df2.to_csv('./data/predict_f2.csv', sep = ',', decimal = '.', index = False)
df3.corr().to_csv('./data/corr.csv', sep = ',', decimal = '.', index = True)
df4.to_csv('./data/stacked_f2.csv', sep = ',', decimal = '.', index = False)

