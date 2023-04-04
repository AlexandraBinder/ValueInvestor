import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


horizontal_division = '-----------------------------------------------------------------------------'

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminate last row with summary data
    df = df.head(df.shape[0]-1)

    # Convert Date to datetime format and organize by date
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%b %d, %Y"))
    df = df.sort_values('Date')

    # Convert Volume data to numeric type
    volume = [float(vol[:-1])*1e3 if 'K' in vol else vol for vol in df['Vol.']]
    df['Vol.'] = [float(vol[:-1])*1e6 if (isinstance(vol, str)) and ('M' in vol) else vol for vol in volume]
    
    # Convert Change % to numeric type
    df['Change %'] = [float(ch[:-1]) if '%' in ch else 0 for ch in df['Change %']]

    # Change Price dtype to float
    df[['Price', 'Open', 'Low', 'High']] = df[['Price', 'Open', 'Low', 'High']].astype("float32")

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

def make_function_stationary(df: pd.DataFrame) -> pd.DataFrame:

    result = adfuller(df['Price'].values, autolag='AIC')
    
    if result[1] > 0.05:
        print(f'\nThe original price values: \nThe time series is NOT stationary and the p-value is {result[1]}')
        print('\nCalculating difference...')
        result = adfuller(np.diff(df['Price'].values), autolag='AIC')
    
    if result[1] < 0.05:
        print(f'\nThe price values after difference: \nThe time series IS stationary and the p-value is {result[1]}')
        difference = df['Price'].diff()
        df['Difference'] = difference
    else:
        print('The time series is NOT stationary, you may need to make another difference')
    
    return df

def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    
    print('- Day, Month, Year\n- Is_quarter_end\n- Low-high\n- Mean and std values for specific time windows (3, 7 and 30 days)')
    # Add Day, Month and Year columns
    df['Day'] = df['Date'].apply(lambda x: x.day)
    df['Month'] = df['Date'].apply(lambda x: x.month)

    # Add Quarter End column
    df['Is_quarter_end'] = np.where(df['Month']%3==0,1,0)
    
    # Add Low-high column
    df['Low-high']  = df['Low'] - df['High']

    # Add Mean and std value for specific time windows
    features = ['Open',	'High', 'Low', 'Vol.', 'Change %']
    windows = [3, 7, 30]
    
    df_rolled_3_days = df[features].rolling(window=windows[0], min_periods=0)
    df_rolled_7_days = df[features].rolling(window=windows[1], min_periods=0)
    df_rolled_30_days = df[features].rolling(window=windows[2], min_periods=0)
    
    df_mean_3d = df_rolled_3_days.mean(numeric_only=True).shift(1).reset_index()
    df_mean_7d = df_rolled_7_days.mean(numeric_only=True).shift(1).reset_index()
    df_mean_30d = df_rolled_30_days.mean(numeric_only=True).shift(1).reset_index()

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
    
    # Remove null values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    return df

def data_analysis(df: pd.DataFrame) -> pd.DataFrame:
    
    df = process_data(df)
    print('\nData with initial processing:')
    display(df.head())

    plot_time_series(df, y=['Price', 'Low', 'High'], 
                    labels=['Original time series', 'Low', 'High'],
                    colors=['black', 'tab:blue', 'tab:orange'])
    
    print(horizontal_division)
    print('Make function stationary:')
    df = make_function_stationary(df)
    
    plot_time_series(df, y=['Difference'], labels=['Difference'], colors=['tab:green'])
    
    print(horizontal_division)
    print('Add additional features:')
    df = add_additional_features(df)
    display(df.head())

    print(horizontal_division)
    print('Final attribute list:\n', df.columns)

    return df

def get_stock_name(stock_file: str) -> str:
    stock_name = stock_file.split('(')[1]
    stock_name = stock_name.split(')')[0]
    stock_country = stock_file.split('-')[2][1:-1]
    print(f'Chosen stock: {stock_name} - {stock_country}')
    return stock_name

def plot_bollinger_bands(df: pd.DataFrame, buy_signal: list, sell_signal: list, hold_signal: list, model_name: str):
    fig, ax = plt.subplots(figsize=(16, 5))
    df['predictions'].plot(label='Predicted prices', linewidth=1.5)
    df['Upper Band'].plot(label='Upper BB', linestyle='--', linewidth=1.5)
    df['rolling_avg'].plot(label='Middle BB', linestyle='--', linewidth=1.5)
    df['Lower Band'].plot(label='Lower BB', linestyle='--', linewidth=1.5)
    
    plt.scatter(df.index, buy_signal, marker='^', color='tab:green', label='Buy', s=100)
    plt.scatter(df.index, np.absolute(sell_signal), marker='v', color='tab:red', label='Sell', s=100)
    plt.scatter(df.index, hold_signal, marker='*', color='tab:blue', label='Hold', s=100)

    plt.title(f'Bollinger Bands Strategy for {model_name} - Trading Signals', fontsize=20)
    plt.legend(loc='upper left')
    plt.show()

def plot_train_val_test(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize = (10,3))
    
    # print(X_train)
    # print(X_train.shape, X_test.shape)
    # print(X_train)
    ax.plot(X_train['Date'], y_train, label="Train", color='tab:blue')
    ax.plot(X_test['Date'], y_test, label="Test", color='tab:orange')
    
    ax.xaxis.set_major_locator(md.MonthLocator())
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
    ax.set_ylabel('Stock price')
    ax.set_xlabel('Date')
    ax.tick_params(labelrotation=45)
    ax.legend()
    plt.show()