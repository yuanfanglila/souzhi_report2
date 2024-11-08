# 大语言模型下的时间序列预测方法

## 前沿
* 时间序列预测是零售、能源、金融、医疗保健、气候科学等各个领域决策的重要组成部分。传统上，预测主要使用 ARIMA 和 ETS 等统计模型。至少在最近转向深度学习技术之前，这些模型一直是可靠的工具。这一转变可归因于大量、多样的时间序列数据源的可用性，以及业务预测问题的出现，这些都发挥了深度预测的优势，即从大量时间序列集合中提取模式的能力。
* 预测下一个标记的语言模型和预测下一个值的时间序列预测模型之间的根本区别是什么？尽管存在明显的区别-来自有限词典的标记与来自无界、通常是连续域的值-但这两种努力从根本上都旨在对数据的顺序结构进行建模以预测未来模式。所以，好的语言模型当然可以预测时间序列吗！

## 方法
* 我们基于Huggingface上的Chronos时间序列模型进行时间序列预测分析，这是一个最低限度适用于时间序列预测的语言建模框架。Chronos通过对实际值进行简单的缩放和量化，将时间序列标记为离散的Token。通过这种方式，我们可以在这种“时间序列语言”上训练现成的语言模型，而无需更改模型架构。
* 我们基于[EPS数据平台](https://www.epsnet.com.cn/index.html#/Index)对12组数据进行了全面评估，测试结果基本超越了传统的ARIMA模型。值得注意的是，基于大语言的Chronos模型实现了令人印象深刻的开箱即用的零样本预测性能，无需针对特定任务进行调整。

## 实验结果
* ![评价指标](https://github.com/yuanfanglila/souzhi_report2/blob/master/image/%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87.jpg)

* ![年度时间序列预测](https://github.com/yuanfanglila/souzhi_report2/blob/master/image/%E5%B9%B4%E5%BA%A6%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B.jpg)

* ![季度时间序列预测](https://github.com/yuanfanglila/souzhi_report2/blob/master/image/%E5%AD%A3%E5%BA%A6%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B.jpg)

* ![两种模型的年度时序预测](https://github.com/yuanfanglila/souzhi_report2/blob/master/%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%B9%B4%E9%A2%84%E6%B5%8B/%E4%BA%BA%E5%9D%87%E5%9B%BD%E5%86%85%E6%80%BB%E4%BA%A7%E5%80%BC%EF%BC%88%E5%85%83%EF%BC%89_%E6%A8%A1%E5%9E%8B%E6%AF%94%E8%BE%83.png)

## 所需环境
* Anaconda3（建议使用）
* python3.6/3.7/3.8
* pycharm (IDE)
* pytorch 2.10 
* transformers
  
