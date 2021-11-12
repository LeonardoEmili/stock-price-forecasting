# Stock Price Forecasting

Neural stock price forecasting system using _fundamental analysis_ and _technical analysis_ to predict the trend of stocks from the S&amp;P 500 index. The main contributions of this work are summarized as follows:
- Develop the first approach with _Pytorch Lightning_ as a learning framework, employing attention and Recurrent Neural Networks (RNNs). For further insights, read the dedicated [report](https://github.com/LeonardoEmili/stock-price-forecasting/releases/download/1.0/report.pdf) or the related [notebook](https://github.com/LeonardoEmili/stock-price-forecasting/blob/main/forecasting.ipynb).
- Develop a distributed approach with _Pytorch_, _PySpark_, and _Petastorm_, leveraging a cluster of nodes to parallelize the computation. It builds on top of the former and extends it introducing the powerful Spark's SQL queries, enabling the system to scale with a large amount of data. For an overview of the system, see the [slides](https://github.com/LeonardoEmili/stock-price-forecasting/releases/download/1.0/presentation.pdf) or the related [notebook](https://github.com/LeonardoEmili/stock-price-forecasting/blob/main/dist_forecasting.ipynb).

## Datasets
We use data from Kaggle's public challenges, namely a [first dataset](https://www.kaggle.com/jerryhans/key-statistics-yahoo-finance-stocks-of-2003-2013) with financial reports from S&amp;P 500 from 2003 to 2013, and a [second dataset](https://www.kaggle.com/paultimothymooney/stock-market-data) containing stock market data. By aligning the two datasets and removing outliers (refer to the notebooks to see how the alignment is performed), we get an enriched dataset that can be used to perform both fundamental and technical analysis.

## Results
A benchmark showing the performance of our trading strategy algorithm (details in the slides, pages 14-16).
<div class="table*">
<table>
<tbody>
<tr class="odd">
<td style="text-align: left;" rowspan="1"></td>
<td style="text-align: center;" colspan="1">MSE</td>
<td style="text-align: center;" colspan="1">R<sup>2</sup></td>
<td style="text-align: center;" colspan="1">Adjusted R<sup>2</sup></td>
<td style="text-align: center;" colspan="1">Operation accuracy</td>
<td style="text-align: center;" colspan="1">Profit</td>
</tr>
<tr class="odd">
<td style="text-align: left;">DecisionTreeRegressor</td>
<td style="text-align: center;">0.078</td>
<td style="text-align: center;">0.852</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">55.45%</td>
<td style="text-align: center;">35.97%</td>
</tr>
<tr class="even">
<td style="text-align: left;">RandomForestRegressor</td>
<td style="text-align: center;">0.104</td>
<td style="text-align: center;">0.803</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;"><strong>57.01%</strong></td>
<td style="text-align: center;">51.61%</td>
</tr>
<tr class="odd">
<td style="text-align: left;">LSTM</td>
<td style="text-align: center;"><strong>0.021</strong></td>
<td style="text-align: center;"><strong>0.939</strong></td>
<td style="text-align: center;"><strong>0.897</strong></td>
<td style="text-align: center;">56.52%</td>
<td style="text-align: center;"><strong>58.35%</strong></td>
</tr>
</tbody>
</table>
</div>

## How to train the distributed system?
In case you would like to install and configure PySpark on your local machine, please follow the instructions described [here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html). Otherwise, you can clone the notebook and import it into Databricks as described [here](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2019135542862542/584054368563718/5339565930708803/latest.html).

## How to test the system?
For a simple and ready-to-use test, simply run the `test/evaluate.py` script that refers to the distributed system with pre-trained weights for the LSTM model. Otherwise, you can re-train the system using a model of your choice, and use the new weights to perform the evaluation.

## Project structure

    .
    ├── data/                     # Stock prices and fundamental data
    ├── report/
    │   ├── main.pdf              # Project report for the dlai-2021 course
    │   ├── main.tex
    │   └── ...
    ├── test/
    │   ├── data/                 # Model weights and test data
    │   ├── evaluate.py           # Evaluation script
    │   └── ...
    ├── dist_forecasting.ipynb    # PySpark distributed stock prediction system
    ├── forecasting.ipynb         # Stock prediction system
    ├── environment.yml           # Training environment
    └── ...
