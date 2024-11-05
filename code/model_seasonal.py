import pandas as pd  # requires: pip install pandas
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, \
    mean_absolute_percentage_error
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np
from pmdarima import auto_arima
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(2024)  # 你可以选择任何整数作为种子

def regression_metrics(true, pred):
    MSE = mean_squared_error(true, pred)
    RMSE = np.sqrt(mean_squared_error(true, pred))
    MAE = mean_absolute_error(true, pred)
    MedianAE = median_absolute_error(true, pred)
    MAPE = mean_absolute_percentage_error(true, pred)

    result = pd.Series(index=['均方误差[MSE]', '均方根误差[RMSE]', '平均绝对误差[MAE]',
                              '绝对误差中位数[MedianAE]', '平均绝对百分比误差[MAPE]'],
                       data=[MSE, RMSE, MAE, MedianAE, MAPE])
    return result


pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.float32,
)

def auto_arima_forecast(data, n_forecast=4):
    model = auto_arima(data, seasonal=False, trace=True,
                       error_action='ignore', suppress_warnings=True)
    model.fit(data)
    forecast = model.predict(n_periods=n_forecast)
    return model, forecast


df1 = pd.read_excel(r"C:\Users\Administrator\Desktop\报告3\周期预测\工业增加值.xlsx")
output_file = r'C:\Users\Administrator\Desktop\报告3\时间序列季度预测\预测结果.txt'
with open(output_file, 'w') as f:
    for column in df1.columns[1:]:
        print(f"processing column :{column}")
        split_index = int(len(df1)*0.80)
        df = df1.iloc[:split_index]

        prediction_length = len(df1) - split_index

        forecast_chronos = pipeline.predict(
            context=torch.tensor(df[column].values),
            prediction_length=prediction_length,
            num_samples=10,
        )

        forecast_index = range(len(df), len(df1))
        low_chronos, median_chronos, high_chronos = np.quantile(forecast_chronos[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        model_arima, forecast_arima = auto_arima_forecast(df[column], n_forecast=prediction_length)

        # 绘制两个模型的预测结果
        plt.figure(figsize=(10, 6))
        plt.plot(df1[column], color="royalblue", label="历史数据")
        plt.plot(forecast_index, median_chronos, color="tomato", label="Chronos预测中位数", marker='x')
        plt.fill_between(forecast_index, low_chronos, high_chronos, color="tomato", alpha=0.3, label="Chronos 90% 预测区间")
        plt.plot(forecast_index, forecast_arima, color="green", label="ARIMA预测", marker='o')
        plt.xlabel('时间', fontsize=14)  # 设置横轴字体大小
        plt.ylabel('值', fontsize=14)  # 设置纵轴字体大小
        plt.legend(fontsize=12)  # 设置图例字体大小
        plt.grid()

        # 设置纵轴为科学计数法
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0e}'))

        plt.savefig(f"C:\\Users\\Administrator\\Desktop\\报告3\\时间序列季度预测\\{column}_模型比较.png")
        plt.close()

        # 计算回归指标
        historical_data = df1[column].iloc[split_index:].values
        result_chronos = regression_metrics(historical_data, median_chronos)
        result_arima = regression_metrics(historical_data, forecast_arima)

        # 将结果写入 TXT 文件
        f.write(f"指标: {column}\n")
        f.write("Chronos模型评估结果:\n")
        f.write(result_chronos.to_string())
        f.write("\n\n")
        f.write("ARIMA模型评估结果:\n")
        f.write(result_arima.to_string())
        f.write("\n\n")
