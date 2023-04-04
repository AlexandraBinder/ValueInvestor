import pandas as pd
import numpy as np
import os
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import torch.optim as optim
import torch.utils.data as data
import analysis


def get_train_test(df: pd.DataFrame, threshold_date: str='2021-01-01') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Take the stationary Price column version
    df['Price'] = df['Difference']

    # Divide train and test according to threshold_date
    train = df.loc[df['Date'] < datetime.strptime(threshold_date, '%Y-%m-%d')]
    test = df.loc[df['Date'] > datetime.strptime(threshold_date, '%Y-%m-%d')]

    # Drop unnecesary columns
    train = train.drop(columns='Difference')
    test = test.drop(columns='Difference')
        
    # train/ test split   
    X_train = train.drop(columns='Price')
    y_train = train['Price']

    X_test = test.drop(columns='Price')
    y_test = test['Price']

    return X_train, y_train, X_test, y_test

def univariate_LSTM(y_train: pd.DataFrame, y_test: pd.DataFrame, performance_report: dict, lookback: int=4):

    # Get univariate dataset for train and test datasets
    uni_X_train, uni_y_train = get_univariate_dataset(y_train, lookback)
    uni_X_test, uni_y_test = get_univariate_dataset(y_test, lookback)
    
    # Generate a validation dataset from the training dataset
    ratio = len(uni_X_train) * 0.9
    uni_X_val = uni_X_train[int(ratio):,:,:]
    uni_X_train = uni_X_train[:int(ratio),:,:]
    uni_y_val = uni_y_train[int(ratio):,:,:]
    uni_y_train = uni_y_train[:int(ratio),:,:]
    
    y_pred_test, test_rmse, train_rmse = train_model(uni_X_train, uni_y_train, uni_X_val, uni_y_val, uni_X_test, uni_y_test)
    model_name = 'univariate_LSTM'

    performance_report['Model'].append(model_name)
    performance_report['Train score'].append(np.float32(train_rmse))
    performance_report['Test score'].append(np.float32(test_rmse))

    return model_name, y_pred_test[:, -1, :].numpy()

def multivariate_LSTM(X_train: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, performance_report: dict, lookback: int=4):
    
    # Generate a validation dataset from the training dataset
    ratio = len(X_train) * 0.9
    multi_X_val = X_train.iloc[int(ratio):]
    multi_X_train = X_train.iloc[:int(ratio)]
    multi_y_val = y_train[int(ratio):]
    multi_y_train = y_train[:int(ratio)]
    
    print("\t(window sample, time steps, features)")
    print('Train:\t', multi_X_train.shape, multi_y_train.shape)
    print('Val:\t', multi_X_val.shape, multi_y_val.shape)
    print('Test:\t', multi_X_test.shape, multi_y_test.shape)

    # Get multivariate dataset for train and test datasets
    multi_X_train, multi_y_train = get_multivariate_dataset(y_train, lookback)
    multi_X_test, multi_y_test = get_multivariate_dataset(y_test, lookback)
    
    y_pred_test, test_rmse, train_rmse = train_model(multi_X_train, multi_y_train, multi_X_val, multi_y_val)
    model_name = 'multivariate_LSTM'

    performance_report['Model'].append(model_name)
    performance_report['Train score'].append(train_rmse)
    performance_report['Test score'].append(test_rmse)

    return performance_report, y_pred_test

def get_univariate_dataset(df, lookback):

    X, y = [], []
    timeseries = np.expand_dims(df.values, 1)

    for i in range(len(timeseries)-lookback):
        feature = timeseries[i : i+lookback]
        target = timeseries[i+1 : i+lookback+1]
        X.append(feature)
        y.append(target)
    
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

def get_multivariate_dataset(df_features, df_target, lookback):
    X, y = [], []
    target = np.expand_dims(df_target.values, 1)
    for i in range(len(df_features)):
        pattern_end = i + lookback
        if pattern_end > len(df_features): break
        seq_x = df_features[i:pattern_end, :]
        seq_y = target[pattern_end-1]
        X.append(seq_x)
        y.append(seq_y)
        
        return torch.tensor(np.array(X)), torch.tensor(np.array(y))
 
class LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, verbose:bool=False):
    
    model = LSTM()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue

        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred_train, y_train))
            y_pred_val = model(X_val)
            val_rmse = np.sqrt(loss_fn(y_pred_val, y_val))

        if verbose:
            print("Epoch %d: train RMSE %.4f, val RMSE %.4f" % (epoch, train_rmse, val_rmse))

    with torch.no_grad():
            y_pred_test = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred_test, y_test))

    return y_pred_test, test_rmse, train_rmse

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

def apply_bollinger_bands_strategy(predictions: list, rate: int, model_name: str):

    import talib as ta
    buy_signal, sell_signal, hold_signal = [], [], []

    df = pd.DataFrame({"predictions": predictions})
    df['predictions'] = df['predictions'].apply(lambda x: x[-1])

    df['rolling_avg'] = df['predictions'].rolling(rate).mean()
    df['rolling_std'] = df['predictions'].rolling(rate).std()
    df['Upper Band']  = df['rolling_avg'] + (df['rolling_std'] * 2)
    df['Lower Band']  = df['rolling_avg'] - (df['rolling_std'] * 2)
    df = df.dropna()
    df = df.reset_index(drop = True)
  
    for idx in df.index:
        
        # Sell if price is higher than the Upper Band
        if df.loc[idx, 'predictions'] > df.loc[idx, 'Upper Band']:  
            buy_signal.append(np.nan)
            sell_signal.append(df.loc[idx, 'predictions'])
            hold_signal.append(np.nan)
        
        # Buy if price is lower than the Lower Band
        elif df.loc[idx, 'predictions'] < df.loc[idx, 'Lower Band']: 
            sell_signal.append(np.nan)
            buy_signal.append(df.loc[idx, 'predictions'])
            hold_signal.append(np.nan)
        
        # Hold if price is between the Upper and Lower band
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
            hold_signal.append(df.loc[idx, 'predictions']) 

    labels_df = pd.DataFrame({"buy_signal": buy_signal, "sell_signal": sell_signal, "hold_signal": hold_signal})
    labels_df.to_csv(os.path.join('final_predictions_with_recommendation', f'{model_name}_predictions.csv'))

    analysis.plot_bollinger_bands(df, buy_signal, sell_signal, hold_signal, model_name)

    return labels_df

def calculate_saved_money(df: pd.DataFrame):

    percentage_list = []
    index_list = [0]
    
    [index_list.append(idx) for idx in df.index if df.loc[idx, 'buy_signal'] or df.loc[idx, 'sell_signal']];
    index_list.append(df.shape[0]-1)

    # If a buy signal or sell signal exists
    if df[df['buy_signal'] == True].shape[0] != 0 or df[df['sell_signal'] == True].shape[0] != 0:
      
        for j in range(len(index_list)):
            if j == len(index_list) - 1: break
            if j == len(index_list) - 3:
                start = index_list[j+1]
                end = index_list[-1]
            else:
                start = index_list[j]
                end = index_list[j+1]
                if j == len(index_list) - 2: break
                
            difference = df.loc[end, 'Price'] - df.loc[start, 'Price'] # last value - first value
            percent = np.round((difference/df.loc[start, 'Price']) * 100, 2)
            percentage_list.append(percent)
        
    # If only hold signal exists
    else:
        difference = df.loc[df.shape[0]-1, 'Price'] - df.loc[0, 'Price'] # last value - first value
        percent = np.round((difference/df.loc[0, 'Price']) * 100, 2)
        percentage_list.append(percent)
        
    
    print(f'Following the model recommendations we are able to save the following percentages of money: {percentage_list}')
    
    return percentage_list

def model_training_report(df: pd.DataFrame):

    # Split data into train and test according to threshold date
    X_train, y_train, X_test, y_test = get_train_test(df, '2021-01-01')
    analysis.plot_train_val_test(X_train, y_train, X_test, y_test)

    # Keep wanted features
    X_train = X_train[['Open', 'High', 'Low', 'Vol.', 'Change %']]
    X_test = X_test[['Open', 'High', 'Low', 'Vol.', 'Change %']]

    # Keep track of model results
    performance_report = {"Model": [], "Train score": [], "Test score": []}
    model_name_uni_LSTM, y_pred_uni_LSTM = univariate_LSTM(y_train, y_test, performance_report)
    # model_name_multi_LSTM, y_pred_multi_LSTM = multivariate_LSTM(y_train, y_test, performance_report)

    predictions = {model_name_uni_LSTM: y_pred_uni_LSTM}

    performance_report = pd.DataFrame(performance_report)

    min_score_model_name = performance_report['Model'][performance_report['Test score'].idxmin()]
    for key in predictions:
        if key == min_score_model_name:
            bb_predictions = predictions[key]

    labels_df = apply_bollinger_bands_strategy(list(bb_predictions), 20, min_score_model_name) 
    labels_df[labels_df.notnull()] = 1
    labels_df = labels_df.fillna(0)

    features = X_test[['Open', 'High', 'Low', 'Vol.', 'Change %']]
    features['Price'] = y_test
    predictions_df = features.tail(labels_df.shape[0])
    predictions_df = predictions_df.reset_index(drop=True)
    predictions_with_label_df = pd.concat([predictions_df, labels_df], axis=1)
    predictions_with_label_df.to_csv(os.path.join('final_predictions_with_recommendation', f'{min_score_model_name}_predictions_with_labels.csv'))
    
    saved_money = calculate_saved_money(predictions_with_label_df)

    display(performance_report)