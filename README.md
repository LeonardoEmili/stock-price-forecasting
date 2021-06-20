# Stock Price Forecasting

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeonardoEmili/stock-price-forecasting/blob/main/forecasting.ipynb)

Stock price forecasting system to predict the trend of stocks from the S&amp;P 500 index.

## How to train the distributed system?
In case you would like to install and configure PySpark on your local machine, please follow the instructions described [here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html). Otherwise, you can clone the notebook and import it into Databricks as described [here](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2019135542862542/584054368563718/5339565930708803/latest.html).

## How to test the system?
For a simple and ready to use test, simply run the `test/evaluate.py` script that refers to the distributed system with pre-trained weights for the LSTM model. Otherwise, you can re-train the system using a model of your choice, and use the new weights to perform the evaluation.

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

## Authors (alfabetical order)

- [Leonardo Emili](https://github.com/LeonardoEmili)
- [Alessio Luciani](https://github.com/AlessioLuciani)
