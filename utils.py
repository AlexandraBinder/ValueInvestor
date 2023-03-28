import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
import torch
import torch.nn as nn


DATE_CONVERSION = {month: index for index, month in enumerate(calendar.month_abbr) if month}

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminate last row with summary data
    df = df.head(df.shape[0]-1)

    # Convert Date to datetime format and organize by date
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%b %d, %Y"))
    df = df.sort_values('Date')

    # Change Price dtype to float
    df[['Price', 'Low', 'High']] = df[['Price', 'Low', 'High']].astype("float32")

    # Drop unnecessary columns, keep date and Price
    df.drop(columns=['Open', 'Vol.', 'Change %'], inplace=True)
    
    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

def plot_time_series(timeseries: pd.DataFrame, y: list=[], labels: list=[], colors: list=[]):
    
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize = (10,3))
    
    for i in range(len(y)):
        ax.plot(timeseries['Date'], timeseries[y[i]], label=labels[i], color=colors[i])
    
    ax.xaxis.set_major_locator(md.MonthLocator())
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
    ax.set_ylabel('Stock price')
    ax.set_xlabel('Date')
    ax.tick_params(labelrotation=45)
    ax.legend()
    plt.show()

def check_if_function_stationary(df):
    """
    This function makes the time series stationary.
    """
    # ADF Test to fins the p-value
    result = adfuller(df['Price'].values, autolag='AIC')
    if result[1] > 0.05:
        print(f'The original price values: \nThe time series is NOT stationary and the p-value is {result[1]}')
        result = adfuller(np.diff(df['Price'].values), autolag='AIC')
    if result[1] < 0.05:
        print(f'\nThe price values after difference: \nThe time series IS stationary and the p-value is {result[1]}')
        difference = df['Price'].diff()
        df['Difference'] = difference
    else:
        print('The time series is NOT stationary, you may need to make another difference')
    return df

def add_additional_features(df):

    features = ['High', 'Low']
    
    # Days to predict
    windows = [3, 7, 30]
    
    # Rolling
    df_rolled_3_days = df[features].rolling(window=windows[0], min_periods=0)
    df_rolled_7_days = df[features].rolling(window=windows[1], min_periods=0)
    df_rolled_30_days = df[features].rolling(window=windows[2], min_periods=0)
    
    # Find the mean
    df_mean_3d = df_rolled_3_days.mean(numeric_only=True).shift(1).reset_index()
    df_mean_7d = df_rolled_7_days.mean(numeric_only=True).shift(1).reset_index()
    df_mean_30d = df_rolled_30_days.mean(numeric_only=True).shift(1).reset_index()

    # Find the std
    df_std_3d = df_rolled_3_days.std(numeric_only=True).shift(1).reset_index()
    df_std_7d = df_rolled_7_days.std(numeric_only=True).shift(1).reset_index()
    df_std_30d = df_rolled_30_days.std(numeric_only=True).shift(1).reset_index()

    for feature in features:
        df[f"{feature}_mean_lag_{windows[0]}"] = df_mean_3d[feature]
        df[f"{feature}_mean_lag_{windows[1]}"] = df_mean_7d[feature]
        df[f"{feature}_mean_lag_{windows[2]}"] = df_mean_30d[feature]

        df[f"{feature}_std_lag_{windows[0]}"] = df_std_3d[feature]
        df[f"{feature}_std_lag_{windows[1]}"] = df_std_7d[feature]
        df[f"{feature}_std_lag_{windows[2]}"] = df_std_30d[feature]
    
    # remove the nulls
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.set_index("Date", drop=False, inplace=True)

    # add the other features
    df["month"] = df['Date'].dt.month
    df["week"] = df['Date'].dt.isocalendar().week
    df["day"] = df['Date'].dt.day
    df["day_of_week"] = df['Date'].dt.dayofweek
    
    return df

def get_train_test(df: pd.DataFrame, columns:list, threshold_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # Take the stationary Price column version
    # df['Price'] = df['Difference']

    train = df.loc[df['Date'] < datetime.strptime(threshold_date, '%Y-%m-%d')]
    test = df.loc[df['Date'] > datetime.strptime(threshold_date, '%Y-%m-%d')]

    return train, test

def create_dataset(df, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    timeseries = np.expand_dims(df['Price'].values, 1)
    for i in range(len(timeseries)-lookback):
        feature = timeseries[i : i+lookback]
        target = timeseries[i+1 : i+lookback+1]
        X.append(feature)
        y.append(target)
    
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))
 
class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def get_RMSE(predicted, actual):
    return np.sqrt(mean_squared_error(actual, predicted))
    
def get_test(model, df: pd.DataFrame, X_train, X_test, lookback) -> pd.DataFrame:
    with torch.no_grad():
        timeseries = np.expand_dims(df['Price'].values, 1)
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:X_train.shape[0]+lookback] = model(X_train)[:, -1, :]

        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[X_train.shape[0]+2*lookback:len(timeseries)] = model(X_test)[:, -1, :]

    return train_plot, test_plot
