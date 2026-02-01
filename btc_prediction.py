# Databricks notebook source
# MAGIC %pip install prophet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text(name="end_date", defaultValue="2026-01-31")
dbutils.widgets.text(name="start_date", defaultValue="2020-01-01")
start_date = dbutils.widgets.get("start_date")
end_date = dbutils.widgets.get("end_date")
# print(end_date)
sql = f"""
               select * from default.bitmex_trade_daily_stats 
               where symbol like '%XBTUSD%' and side = 'Sell'
               and to_date(date) > '{start_date}' and to_date(date) < '{end_date}'
               order by to_date(date) desc
               """
# print(sql)
df = spark.sql(sql)
display(df)

# COMMAND ----------

import pandas as pd
from prophet import Prophet
from pyspark.sql.functions import to_date, year, col

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType, to_date, year, col, current_date
filtered_df = (
    df.withColumn('ds', to_date(col('date')))
        .withColumn('y',col('avg_price')).select('ds','y')
)

# COMMAND ----------

pdf = filtered_df.toPandas()

# COMMAND ----------

specific_dates = {'ds': ['2012-11-28','2016-07-09','2020-05-11','2024-04-19', '2028-04-11'], 
                  'holiday': ['btc_halving','btc_halving','btc_halving','btc_halving', 'btc_halving']}
                  # Include fomc_rate_cut, fomc_rate_hold, fomc_rate_hike
holiday = pd.DataFrame.from_dict(specific_dates)
holiday['ds'] = pd.to_datetime(holiday['ds'])

# COMMAND ----------

# configure the model
model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    holidays=holiday
)
# train the model
model.fit(pdf)
# BUILD FORECAST AS BEFORE
# --------------------------------------
# make predictions
future = model.make_future_dataframe(periods=240, freq="d", include_history=True)
future = model.predict(future)
# Plot the forecast with increased figure size
fig1 = model.plot(future)
fig1.set_size_inches(30, 7)  # Increase the figure size
# Plot the forecast components with increased figure size
fig2 = model.plot_components(future)
fig2.set_size_inches(15, 10)  # Increase the figure size
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Access the axes of the plot
ax = fig1.gca()
# Set major ticks format
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show a tick every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
# Rotate labels to avoid overlap
for label in ax.get_xticklabels():
    label.set_rotation(90)  # Rotate labels by 45 degrees for better readability
# Rotate labels to avoid overlap
plt.xticks(rotation=90)  # Rotate labels by 45 degrees for better readability
# Show the plot with adjustments
plt.show()

# COMMAND ----------

# DONT TOUCH
# configure the model
model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    holidays=holiday
)
# train the model
model.fit(pdf)
# BUILD FORECAST AS BEFORE
# --------------------------------------
# make predictions
future = model.make_future_dataframe(periods=240, freq="d", include_history=True)
future = model.predict(future)
# Plot the forecast with increased figure size
fig1 = model.plot(future)
fig1.set_size_inches(30, 7)  # Increase the figure size
# Plot the forecast components with increased figure size
fig2 = model.plot_components(future)
fig2.set_size_inches(15, 10)  # Increase the figure size
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Access the axes of the plot
ax = fig1.gca()
# Set major ticks format
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show a tick every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
# Rotate labels to avoid overlap
for label in ax.get_xticklabels():
    label.set_rotation(90)  # Rotate labels by 45 degrees for better readability
# Rotate labels to avoid overlap
plt.xticks(rotation=90)  # Rotate labels by 45 degrees for better readability
# Show the plot with adjustments
plt.show()
# DONT TOUCH

# COMMAND ----------

# Changepoint range 0.9
from datetime import datetime
from matplotlib.ticker import MaxNLocator
# Convert string dates to datetime objects
dates = ['2024-04-19']
events = ["4th Halving"]
dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
# configure the model
model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    holidays=holiday,
    # testing below
    changepoint_range = 0.8, #(0.9)
    # testing 2
    changepoint_prior_scale=0.06,
    n_changepoints=250,
)
# train the model
model.fit(pdf)
# BUILD FORECAST AS BEFORE
# --------------------------------------
# make predictions
future = model.make_future_dataframe(periods=240, freq="d", include_history=True)
future = model.predict(future)
#########################################
# Calculate a 30-day moving average of the historical data
# Reverse the dataframe
pdf_reversed = pdf.iloc[::-1].reset_index(drop=True)
# Calculate the moving averages from the bottom up
pdf_reversed['y_moving_avg_7'] = pdf_reversed['y'].rolling(window=7).mean()
pdf_reversed['y_moving_avg_14'] = pdf_reversed['y'].rolling(window=14).mean()
pdf_reversed['y_moving_avg_28'] = pdf_reversed['y'].rolling(window=28).mean()
pdf_reversed['y_moving_avg_90'] = pdf_reversed['y'].rolling(window=90).mean()
# Reverse the dataframe back to the original order
pdf['y_moving_avg_7'] = pdf_reversed['y_moving_avg_7'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_14'] = pdf_reversed['y_moving_avg_14'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_28'] = pdf_reversed['y_moving_avg_28'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_90'] = pdf_reversed['y_moving_avg_90'].iloc[::-1].reset_index(drop=True)
##########################################
# Plot the forecast with increased figure size
fig1 = model.plot(future)
fig1.set_size_inches(30, 20)  # Increase the figure size
# Plot the forecast components with increased figure size
fig2 = model.plot_components(future)
fig2.set_size_inches(15, 10)  # Increase the figure size
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Access the axes of the plot
ax = fig1.gca()
# Add vertical lines for each halving date
for date, event in zip(dates, events):
    ax.axvline(x=date, color='red', linestyle='--', lw=2)  # Add a red dashed vertical line
    ax.text(date, ax.get_ylim()[1], event, rotation=45, verticalalignment='bottom')  # Annotate the line
# Set major ticks format
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show a tick every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-01'))  # Format as Year-Month
# Rotate labels to avoid overlap
for label in ax.get_xticklabels():
    label.set_rotation(90)  # Rotate labels by 45 degrees for better readability
# Add more y-axis ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=50))  # Set a maximum of 10 y-axis ticks
# Add horizontal shaded regions to alternate the color bars
for i in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1]), 10000):
    if i % 20000 == 0:  # Alternating every 20,000 units on the y-axis
        ax.axhspan(i, i + 10000, facecolor='lightgray', alpha=0.5)
# Rotate labels to avoid overlap
plt.xticks(rotation=90)  # Rotate labels by 45 degrees for better readability
ax.set_xlim([datetime(2023, 1, 1), future['ds'].max()])  # Limit x-axis to show data from 2020 onwards
###
# Plot the 7-day moving average
ax.plot(pdf['ds'], pdf['y_moving_avg_7'], color='green', label='7-Day Moving Average')
# Plot the 14-day moving average
ax.plot(pdf['ds'], pdf['y_moving_avg_14'], color='red', label='14-Day Moving Average')
# Plot the 28-day moving average
ax.plot(pdf['ds'], pdf['y_moving_avg_28'], color='purple', label='28-Day Moving Average')
# Plot the 90-day moving average
ax.plot(pdf['ds'], pdf['y_moving_avg_90'], color='orange', label='90-Day Moving Average')
###
# Show the plot with adjustments
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecast Moving Averages

# COMMAND ----------

# DBTITLE 1,Cell 11
# With tuned values
# DONT CHANGE
from datetime import datetime
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Convert string dates to datetime objects
dates = ['2024-04-19']
events = ["4th Halving"]
dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
# configure the model
model = Prophet(
    interval_width=0.95, # Default 0.8
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    holidays=holiday,
    changepoint_prior_scale=0.1,
    changepoint_range=0.8,
    seasonality_prior_scale=10,
    n_changepoints=300,
)
# {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'n_changepoints': 300, 'changepoint_range': 0.9}
# train the model
model.fit(pdf)
# Make predictions
future = model.make_future_dataframe(periods=240, freq="d", include_history=True)
forecast = model.predict(future)
#########################################
# Calculate a 30-day moving average of the historical data
# Reverse the dataframe
pdf_reversed = pdf.iloc[::-1].reset_index(drop=True)
# Calculate the moving averages from the bottom up
pdf_reversed['y_moving_avg_7'] = pdf_reversed['y'].rolling(window=7).mean()
pdf_reversed['y_moving_avg_14'] = pdf_reversed['y'].rolling(window=14).mean()
pdf_reversed['y_moving_avg_28'] = pdf_reversed['y'].rolling(window=28).mean()
pdf_reversed['y_moving_avg_90'] = pdf_reversed['y'].rolling(window=90).mean()
# Reverse the dataframe back to the original order
pdf['y_moving_avg_7'] = pdf_reversed['y_moving_avg_7'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_14'] = pdf_reversed['y_moving_avg_14'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_28'] = pdf_reversed['y_moving_avg_28'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_90'] = pdf_reversed['y_moving_avg_90'].iloc[::-1].reset_index(drop=True)
##########################################
# Calculate moving averages for the forecasted data
forecast['y_moving_avg_7'] = forecast['yhat'].rolling(window=7).mean()
forecast['y_moving_avg_14'] = forecast['yhat'].rolling(window=14).mean()
forecast['y_moving_avg_28'] = forecast['yhat'].rolling(window=28).mean()
forecast['y_moving_avg_90'] = forecast['yhat'].rolling(window=90).mean()
##########################################
# Plot the forecast with increased figure size
fig1 = model.plot(forecast)
fig1.set_size_inches(30, 20)  # Increase the figure size
# Plot the forecast components with increased figure size
fig2 = model.plot_components(forecast)
fig2.set_size_inches(15, 10)  # Increase the figure size
# Access the axes of the plot
ax = fig1.gca()
# Add vertical lines for each halving date
for date, event in zip(dates, events):
    ax.axvline(x=date, color='red', linestyle='--', lw=2)  # Add a red dashed vertical line
    ax.text(date, ax.get_ylim()[1], event, rotation=45, verticalalignment='bottom')  # Annotate the line
# Set major ticks format
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show a tick every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-01'))  # Format as Year-Month
# Rotate labels to avoid overlap
for label in ax.get_xticklabels():
    label.set_rotation(90)  # Rotate labels by 90 degrees for better readability
# Add more y-axis ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=50))  # Set a maximum of 50 y-axis ticks
# Rotate labels to avoid overlap
plt.xticks(rotation=90)  # Rotate labels by 90 degrees for better readability
ax.set_xlim([datetime(2023, 1, 1), forecast['ds'].max()])  # Limit x-axis to show data from 2023 onwards
###
# Plot the historical moving averages
ax.plot(pdf['ds'], pdf['y_moving_avg_7'], color='green', label='7-Day Moving Average (Historical)')
ax.plot(pdf['ds'], pdf['y_moving_avg_14'], color='red', label='14-Day Moving Average (Historical)')
ax.plot(pdf['ds'], pdf['y_moving_avg_28'], color='purple', label='28-Day Moving Average (Historical)')
ax.plot(pdf['ds'], pdf['y_moving_avg_90'], color='orange', label='90-Day Moving Average (Historical)')
# Plot the forecasted moving averages
ax.plot(forecast['ds'], forecast['y_moving_avg_7'], color='green', linestyle='--', label='7-Day Moving Average (Forecast)')
ax.plot(forecast['ds'], forecast['y_moving_avg_14'], color='red', linestyle='--', label='14-Day Moving Average (Forecast)')
ax.plot(forecast['ds'], forecast['y_moving_avg_28'], color='purple', linestyle='--', label='28-Day Moving Average (Forecast)')
ax.plot(forecast['ds'], forecast['y_moving_avg_90'], color='orange', linestyle='--', label='90-Day Moving Average (Forecast)')
# Add a legend
ax.legend()
# Show the plot with adjustments
plt.show()

# COMMAND ----------

# Export historical data (pdf) to CSV
pdf.to_csv("historical_data.csv", index=False)
# Export forecasted data to CSV
forecast.to_csv("forecasted_data.csv", index=False)

# COMMAND ----------

display(pdf)

# COMMAND ----------

display(forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC # Experimenting

# COMMAND ----------

import pandas as pd

halving_dates = pd.to_datetime(['2012-11-28','2016-07-09','2020-05-11','2024-04-19','2028-04-11'])

pre_up_days = 180
post_down_days = 120
post_up_days = 365   # <-- choose how long the "then up" lasts

holiday = pd.concat([
    # Pre: +180d leading into the halving (h-180 ... h-1)
    pd.DataFrame({
        "holiday": "halving_pre_up",
        "ds": halving_dates - pd.Timedelta(days=1),
        "lower_window": -(pre_up_days - 1),   # -179
        "upper_window": 0,                    # up to ds itself (h-1)
    }),

    # Post: -120d immediately after (h ... h+119)
    pd.DataFrame({
        "holiday": "halving_post_down",
        "ds": halving_dates,
        "lower_window": 0,
        "upper_window": post_down_days - 1,   # 119
    }),

    # Post: + again starting day 120 after (h+120 ... h+120+post_up_days-1)
    pd.DataFrame({
        "holiday": "halving_post_up",
        "ds": halving_dates + pd.Timedelta(days=post_down_days),
        "lower_window": 0,
        "upper_window": post_up_days - 1,
    }),
], ignore_index=True)

# COMMAND ----------

# With tuned values
from datetime import datetime
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Convert string dates to datetime objects
dates = ['2024-04-19']
events = ["4th Halving"]
dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
# configure the model
model = Prophet(
    interval_width=0.95, # Default 0.8
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    holidays=holiday,
    changepoint_prior_scale=0.1,
    changepoint_range=0.8,
    seasonality_prior_scale=10,
    n_changepoints=300,
)
# {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'n_changepoints': 300, 'changepoint_range': 0.9}
# train the model
model.fit(pdf)
# Make predictions
future = model.make_future_dataframe(periods=240, freq="d", include_history=True)
forecast = model.predict(future)
#########################################
# Calculate a 30-day moving average of the historical data
# Reverse the dataframe
pdf_reversed = pdf.iloc[::-1].reset_index(drop=True)
# Calculate the moving averages from the bottom up
pdf_reversed['y_moving_avg_7'] = pdf_reversed['y'].rolling(window=7).mean()
pdf_reversed['y_moving_avg_14'] = pdf_reversed['y'].rolling(window=14).mean()
pdf_reversed['y_moving_avg_28'] = pdf_reversed['y'].rolling(window=28).mean()
pdf_reversed['y_moving_avg_90'] = pdf_reversed['y'].rolling(window=90).mean()
# Reverse the dataframe back to the original order
pdf['y_moving_avg_7'] = pdf_reversed['y_moving_avg_7'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_14'] = pdf_reversed['y_moving_avg_14'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_28'] = pdf_reversed['y_moving_avg_28'].iloc[::-1].reset_index(drop=True)
pdf['y_moving_avg_90'] = pdf_reversed['y_moving_avg_90'].iloc[::-1].reset_index(drop=True)
##########################################
# Calculate moving averages for the forecasted data
forecast['y_moving_avg_7'] = forecast['yhat'].rolling(window=7).mean()
forecast['y_moving_avg_14'] = forecast['yhat'].rolling(window=14).mean()
forecast['y_moving_avg_28'] = forecast['yhat'].rolling(window=28).mean()
forecast['y_moving_avg_90'] = forecast['yhat'].rolling(window=90).mean()
##########################################
# Plot the forecast with increased figure size
fig1 = model.plot(forecast)
fig1.set_size_inches(30, 20)  # Increase the figure size
# Plot the forecast components with increased figure size
fig2 = model.plot_components(forecast)
fig2.set_size_inches(15, 10)  # Increase the figure size
# Access the axes of the plot
ax = fig1.gca()
# Add vertical lines for each halving date
for date, event in zip(dates, events):
    ax.axvline(x=date, color='red', linestyle='--', lw=2)  # Add a red dashed vertical line
    ax.text(date, ax.get_ylim()[1], event, rotation=45, verticalalignment='bottom')  # Annotate the line
# Set major ticks format
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show a tick every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-01'))  # Format as Year-Month
# Rotate labels to avoid overlap
for label in ax.get_xticklabels():
    label.set_rotation(90)  # Rotate labels by 90 degrees for better readability
# Add more y-axis ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=50))  # Set a maximum of 50 y-axis ticks
# Rotate labels to avoid overlap
plt.xticks(rotation=90)  # Rotate labels by 90 degrees for better readability
ax.set_xlim([datetime(2023, 1, 1), forecast['ds'].max()])  # Limit x-axis to show data from 2023 onwards
###
# Plot the historical moving averages
ax.plot(pdf['ds'], pdf['y_moving_avg_7'], color='green', label='7-Day Moving Average (Historical)')
ax.plot(pdf['ds'], pdf['y_moving_avg_14'], color='red', label='14-Day Moving Average (Historical)')
ax.plot(pdf['ds'], pdf['y_moving_avg_28'], color='purple', label='28-Day Moving Average (Historical)')
ax.plot(pdf['ds'], pdf['y_moving_avg_90'], color='orange', label='90-Day Moving Average (Historical)')
# Plot the forecasted moving averages
ax.plot(forecast['ds'], forecast['y_moving_avg_7'], color='green', linestyle='--', label='7-Day Moving Average (Forecast)')
ax.plot(forecast['ds'], forecast['y_moving_avg_14'], color='red', linestyle='--', label='14-Day Moving Average (Forecast)')
ax.plot(forecast['ds'], forecast['y_moving_avg_28'], color='purple', linestyle='--', label='28-Day Moving Average (Forecast)')
ax.plot(forecast['ds'], forecast['y_moving_avg_90'], color='orange', linestyle='--', label='90-Day Moving Average (Forecast)')
# Add a legend
ax.legend()
# Show the plot with adjustments
plt.show()

# COMMAND ----------

    "lower_window": [-180, -180, -180, -180],
    "upper_window": [365, 365, 365, 365],