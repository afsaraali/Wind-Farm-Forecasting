# Wind Farm Timeseries Forecasting

- [Background](#background)
- [Data](#datasets-provided)
- [Problem statement](#problem-statement)
- [Exploratory data analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Next steps](#next-steps)

## Background

Wind energy is the most prominent form of renewable energy in the US today, and it will only continue to grow. There are many advantages and disadvantages to relying on wind farming, so this project aims to help stake-holders and non-technical audiences not familiar with this energy source to be more educated on how energy ouputs can be predicted and measured, and to see if the investment in wind is justified/should be increased.


### Datasets Provided

This project is based on this <a href="https://www.kaggle.com/c/GEF2012-wind-forecasting#description">Kaggle dataset</a>. The 'train.csv' contains the wind power measurements from 7 wind farms in hourly increments. Power output is normalized between 0 and 1 so that the wind farms are anonymized. There is also a wind forecast dataset for each wind farm, which contains 2 day ahead hourly forecasts for wind speed and direction. Forecasts are provided in 12 hour increments. The 'benchmark.csv' provides a persistence forecast that serves as a baseline to the models I generated.

The libraries that were used to complete this project are:

- pandas
- numpy
- datetime`
- itertools
- time
- tqdm
- statsmodels
- sklearn
- keras
- tensorflow
- matplotlib
- seaborn
- ipywidgets


### Problem statement

Wind farms produce variable outputs since they are dependent on weather conditions, location, and other logistic factors. It is a slippery slope to balance supply and demand since underproducing and overproducing energy could lead to blackouts and dangerous power surges, respectively.

I used timeseries and machine learning to create 2 day long predictions in hourly increments of power output for a single wind farm. . I measured the accuracy of my models and forecasts by generating the RMSE of each prediction. These scores were compared to the baseline, which was the persistence forecast.

This is what I have achieved so far:

1. Created a univariate ARIMA model to predict power output at a single wind farm
2. Created a univariate LSTM model to predict power output at a single wind farm
3. Plotted both model predictions for a single wind farm.
4. Deployed interactive apps that show said plots locally through voila, ngrok, and Heroku.

The project will be expanded upon to implement my models on all 7 wind farms in the datasets, and to deploy the apps remotely.

## Exploratory data analysis

#### Exploring data layout

<a href="EDA & Model notebooks/1.01_Cleaning_&_EDA.ipynb">EDA and cleaning notebook</a>

The power output dataset is structured so that the 18 month period between 2009/07/01 and 2010/12/31 can be used for model identification and training, and the 18 month period of remaining data is the hold-out test set. The hold-out set has a repeating pattern of 36 hours of available data, followed by 48 hours of missing data which I was able to forecast. 

I verified that there were no gaps in the data. I also carved out a 4 month alidation test set from the training data that followed the same format as the hold-out set. I also verified that the power output data was normalized.

#### Exploring patterns

<a href="EDA & Model notebooks/1.02-EDA-WP1.ipynb">EDA patterns notebook</a>

The time series plot for the first wind farm suggested that the power data was stationary. I was able to ensure that there were no trends or seasonality. I confirmed this by running a Dickey-Fuller test, which produced a p-value so low that I was able to reject the null hypothesis.

## Modeling

### Univariate ARIMA model

<a href="EDA & Model notebooks/2.01-Timeseries_Models_WP1.ipynb">Univariate time series model notebook</a>

I created a univariate ARIMA model for the first wind farm (WP1). The three parameters I had to account for were:

1. **AR (auto-regression)**: This uses historical data as features to predict the next time point. The parameter p determines how many previous timestamps to use in each prediction (aka lags).
2. **I (integrated)**: This parameter was ignored since the data did not require differencing.
3. **MA (moving-average)**: This uses errors in previous predictions to predict the next timestamp. The parameter q determines how many errors are used.

To determine which parameters needed to be tuned, I looked at plots of the autocorrelation function and the partial autocorrelation function. The ACF describes how each data point is correlated to the data point k steps prior, where k is known as the lag. The PACF is similar, but removes the effect of intermediate lags.

The PACF plot indicated that an AR2 model was appropriate, since it cuts off dramatically after 2 lags, and the ACF reduces steadily over time. The AR2 model had an RMSE of 0.27, which was a **~4% improvement on the baseline RMSE** of 0.31 from the persistence forecast method.

I instantiated a grid search to find the optimal AIC, BIC, and RMSE parameter values to see if a better model could be produced.

- **AIC/BIC**: These measure the log-likelihood of observing the data given the model, so it penalizes high numbers of parameters. AIC is likely to find an overfit model, while BIC is likely to find an underfit model.

- **RMSE**: The lower the RMSE, the better the model is at forecasting closely to the target data.

The AIC/BIC optimization indicated an ARIMA model was best, but the RMSE optimization indicated an AR1 model was best. Checking out the model diagnostics of the AR1 model showed that the there was still a bit of a correlation between lagged residuals, so all dependencies in the data were not accounted for. This might mean that the model was overfit to the validation test set. The ARIMA model showed a small improvement on the AR2 model, but it was too miniscule to justify switching over.

I created a widget which is only fully interactive once the notebook is run with voila, but here is a sample output:

<p align="center">
  <img width="800" height="300" src="app showcase/AR_vs_persistence_forecast.png">
</p>

There was a ~4% improvement from the persistence forecast method. I moved on to use machine learning to see if it would generate a better model.

### Univariate LSTM

<a href="EDA & Model notebooks/2.02-LSTM_Model_WP1.ipynb">Univariate LSTM notebook</a>

I also created a univariate LSTM. LSTMs are an ideal form of RNNs to use for timeseries modeling.

My model had one LSTM layer and a dense output layer with default activation of tanh. I tuned these parameters:

- The number of time steps
- The number of nodes in the output layer so that 48 hour predictions could be generated
- Statefulness
- The number of nodes
- The number of epochs

Tuning taught me that:

- Models that use 36 hours of data are not much better than those that use only 2 hours.
- Generating a 2 day forecast in one step creates more accurate forecasts than predicting 1 hour at a time.
- Stateful models perform slightly worse
- Increasing nodes decreased model accuracy
- The ideal number of epochs is 4 max.

The model built from the optimal parameter set produced an RMSE of 0.21, which was a **~10% improvement on the baseline RMSE.**

I created a widget which is only fully interactive once the notebook is run with voila, but here is a sample output:

<p align="center">
  <img width="800" height="300" src="app showcase/LSTM_vs_AR_Forecast.png">
</p>

Of the two models, LSTM is clearly better with an additional reduction in RMSE for the forecasts when compared to the persistence forecast.

## Next steps

The LSTM model is especially promising, but the AR model could be even more improved.

- ARIMA had abnormal residuals. The confidence intervals should be adjusted to account for the higher kurtosis found.

- I can generate a model that outputs a full forecast distribution or attempt Monte Carlo simulation.

- I can tune other hyper-parameters of LSTM like batch size, which I ignored in these models.

- I can incorporate weather wind forecasts, which would explain some of the seasonality I saw for certain time frames even though the data is stationary.

- I can implement a more robust LSTM with added layers to all of the wind farms in the datasets.

-  I can deploy my apps remotely. They are hosted locally through voila and ngrok as of now.

- I can implement a more robust LSTM with added layers to all of the wind farms in the datasets.
