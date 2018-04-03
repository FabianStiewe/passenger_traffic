import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA

### Read input data ###

cwd = os.getcwd()
input_file = os.path.join(cwd, 'input00.txt')

with open(input_file) as f:
    N = int(f.readline())

df = pd.read_csv(input_file, sep='\t', skiprows=1, header=None,
                 names=('passengers',), index_col=0)
df.index = df.index.map(lambda month: int(month[9:]))

### Normalize data for processing to avoid overflows ###

SUM = df['passengers'].sum()

df['normalized'] = df['passengers'] / SUM

### Perform Dickey-Fuller test for stationarity ###

def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic',
                                'p-value',
                                '#Lags Used',
                                'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(df['normalized'])

# p-value is already below 1% but can still be reduced by decomposing
# time series and only modeling the residuals.

### Decompose ###

decomposition = seasonal_decompose(np.asarray(df['normalized']),
                                   model='additive',
                                   freq=12)

df['trend'] = decomposition.trend
df['seasonal'] = decomposition.seasonal
df['residual'] = decomposition.resid

ts_residual = df['residual'].dropna()
test_stationarity(ts_residual)

# p-value has become distinctly smaller.

### Perform linear regression on trend ###

trend = df['trend'].dropna()

x = trend.index.values.reshape(len(trend), 1)
y = trend.values.reshape(len(trend), 1)
regr = LinearRegression()
regr.fit(x, y)
trend_extrapol = regr.predict(np.arange(1,N+13).reshape(N+12,1))


df_ep = pd.DataFrame({'extrapol': trend_extrapol.ravel()},
                     index=np.arange(1,N+13))

df = pd.concat([df, df_ep], axis=1)

### Periodic continuation of seasonal cycle ###

df.loc[N+1:, 'seasonal'] = df.loc[1:12, 'seasonal'].values

### Determine parameters for ARIMA model ###

last_index = ts_residual.index[-1]

lag_acf = acf(ts_residual, nlags=20)
lag_pacf = pacf(ts_residual, nlags=20, method='ols')

confidence_interval = 1.96/np.sqrt(len(ts_residual))

p = min(*np.where(lag_pacf<confidence_interval))
q = min(*np.where(lag_acf<confidence_interval))

### Fit ARIMA model and forecast_residuals ###

to_go = N - last_index + 12
model = ARIMA(np.asarray(ts_residual), order=(p,1,q))
results = model.fit(disp=0)
pred_residuals = results.forecast(steps=to_go)[0]

df_pr = pd.DataFrame({'pred res': pred_residuals},
                     index=np.arange(last_index+1, N+13))
df = pd.concat([df, df_pr], axis=1)
print(df)

### Final results ###

prediction = (df.loc[N+1:, 'extrapol'].values
              + df.loc[N+1:, 'seasonal'].values
              + df.loc[N+1:, 'pred res'].values) * SUM

prediction = prediction.astype(int)
for i in prediction:
    print(i)

### Plot results ###

plt.plot(df['normalized'], color='blue')
plt.plot(df['trend'], color='red')
plt.plot(df['extrapol'], color='green')
plt.plot(df['seasonal'], color='cyan')
plt.plot(df['residual'], color='magenta')
plt.plot(df['pred res'], color='orange')
plt.legend()
plt.show()
