import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# read dataset from csv file

df = pd.read_csv(r'TvMarketing.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3,
                                                    random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Marketing Budget vs Sales (Training Set)')
plt.xlabel('Budget')
plt.ylabel('Sales')
plt.show()

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, regressor.predict(x_test), color='red')
plt.title('Marketing Budget vs Sales (Testing Set)')
plt.xlabel('Budget')
plt.ylabel('Sales')
plt.show()

y_pred = regressor.predict(x_test)
beta0 = regressor.intercept_
beta1 = regressor.coef_
y_pred_full = x * beta1 + beta0
y_mean = np.mean(y)
sst = np.sum((y - y_mean) ** 2)
sse = np.sum((y - y_pred_full) ** 2)
ssr = np.sum((y_pred_full - y_mean) ** 2)
print(f'SST: {sst}\nSSE: {sse}\nSSR: {ssr}\n')

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Square Error: {mse}\nR^2 Error: {r2}\n')
