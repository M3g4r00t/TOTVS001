import json
from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

'''Read and parse data'''
data = json.load(open('sample.txt'))
df = pd.DataFrame()

total_value = []
date = []

for sale in tqdm(data):
    total_value.append(sale['complemento']['valorTotal'])
    date.append(sale['ide']['dhEmi']['$date'])

df['date'] = date
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['total_value'] = total_value

df = df.set_index(['date'])

df = df['total_value'].resample('D').sum()
df = pd.DataFrame(df)
df = df.fillna(0)

df.to_csv('sample.csv')

'''Analyze stationary behavior'''

plt.plot(df)
plt.show()

'''Extract relevant feature'''

df = pd.read_csv('sample.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].dt.dayofweek

X = df['date'].values.reshape(-1, 1)
y = df['total_value']

'''Apply regression tree model'''

regr = DecisionTreeRegressor(max_depth=2)
regr.fit(X, y)

'''Predict'''
X_1 = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_1 = regr.predict(X_1)

'''Show results'''
plt.figure()
plt.plot(X_1, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
