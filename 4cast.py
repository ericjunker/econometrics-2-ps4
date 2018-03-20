#!/usr/bin/env python3
# -*- coding: utf-8 -*-

	
from fbprophet import Prophet
import numpy as np
import pandas as pd

employment_df = pd.read_csv('PAYEMS.csv')
#Rename column headers
employment_df['y'] = employment_df['PAYEMS']
employment_df['ds'] = employment_df['DATE']

employment_df['y_orig'] = employment_df['y']
#take the log of PAYEMS to make it stationary
employment_df['y'] = np.log(employment_df['y'])

#instantiate Prophet with a 95% confidence interval and give it my data
model = Prophet(interval_width=0.95)
model.fit(employment_df)

#predict one month out
future_data = model.make_future_dataframe(periods=1, freq='m')
forecast_data = model.predict(future_data)
#get the last bit of forecast data
print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
#graph log forecast
model.plot(forecast_data)
#model.plot_components(forecast_data)
#get non-log output

print(np.exp(forecast_data[[ 'yhat', 'yhat_lower', 'yhat_upper']]).tail())