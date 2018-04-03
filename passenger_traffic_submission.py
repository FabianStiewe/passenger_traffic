import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

### Read input data ###

N = int(input())

ind = np.zeros(N, dtype=int)
passengers = np.zeros(N, dtype=int)

for i in range(N):
    line = input()
    mo, pa = line.split('\t')
    ind[i] = int(mo[9:])
    passengers[i] = pa

df = pd.DataFrame({'passengers': passengers}, index=ind)

### Normalize data for processing to avoid overflows ###

SUM = df['passengers'].sum()

df['normalized'] = df['passengers'] / SUM

### Decompose ###

decomposition = seasonal_decompose(np.asarray(df['normalized']),
                                   model='additive',
                                   freq=12)

df['trend'] = decomposition.trend
df['seasonal'] = decomposition.seasonal
df['residual'] = decomposition.resid

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

ts_residual = df['residual'].dropna()
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

### Final results ###

prediction = (df.loc[N+1:, 'extrapol'].values
              + df.loc[N+1:, 'seasonal'].values
              + df.loc[N+1:, 'pred res'].values) * SUM

prediction = prediction.astype(int)

for i in prediction:
    print(i)
