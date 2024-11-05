# -*- coding: utf-8 -*-
# pip --default-timeout=1000 install pmdarima

import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa import stattools
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, \
    mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def regression_metrics(true, pred):
    MSE = mean_squared_error(true, pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(true, pred)
    MedianAE = median_absolute_error(true, pred)
    MAPE = mean_absolute_percentage_error(true, pred)

    result = pd.Series(index=['均方误差[MSE]', '均方根误差[RMSE]', '平均绝对误差[MAE]',
                              '绝对误差中位数[MedianAE]', '平均绝对百分比误差[MAPE]'],
                       data=[MSE, RMSE, MAE, MedianAE, MAPE])
    return result


def arima(data, x, n, automatic=True, p=None, d=None, q=None):
    data1 = data[x].dropna()

    if automatic:
        model = pm.auto_arima(data1, start_p=1, start_q=1,
                              test='adf', max_p=10, max_q=10,
                              seasonal=False, trace=True,
                              error_action='ignore', suppress_warnings=True,
                              stepwise=True)
        (p, d, q) = model.order

    log_diff = np.log(data1).diff(4)
    log_diff = log_diff.dropna()
    log_diff.plot()
    from statsmodels.tsa.stattools import adfuller
    adf = adfuller(log_diff)
    plot_acf(log_diff)
    plot_pacf(log_diff)
    plt.show()


    model = sm.tsa.ARIMA(data1, order=(p, d, q))
    result = model.fit()

    # Get the forecast for the next n periods
    forecast = result.get_forecast(steps=n)
    forecast_df = forecast.summary_frame()
    predicted_values = forecast_df['mean']

    return result, predicted_values


# Load data
data = pd.read_excel(r"C:\Users\Administrator\Desktop\报告3\周期预测\人均可支配.xlsx")
x = '城镇居民家庭人均可支配收入（元/人）'
# Determine the split index for 80% training data
split_index = int(len(data) * 0.8)

# Split data into training and testing sets
train_data = data.iloc[:split_index]  # First 80% samples
test_data = data.iloc[split_index:]  # Last 20% samples

# Fit ARIMA model on the training data
model, predictions = arima(train_data, x=[x], n=len(test_data), d=4, p=1, q=1)

# Get actual values from the test set for comparison
actual_values = test_data[x].dropna()

# Calculate regression metrics
metrics = regression_metrics(actual_values, predictions)

# Output predictions and regression metrics
print('Predictions:', predictions.values)
print('Actual Values:', actual_values.values)
print('Regression Metrics:', metrics)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data[x], label='训练集真实值', color='blue')
plt.plot(test_data.index, actual_values, label='测试集真实值', color='green', marker='o')
plt.plot(test_data.index, predictions, label='预测值', color='red', marker='x')
plt.title('ARIMA 真实值与预测值对比')
plt.xlabel('时间')
plt.ylabel(x)
plt.legend()
plt.grid()
plt.show()