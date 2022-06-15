from plotting import plot_data

from sklearn.preprocessing import MinMaxScaler
from convert_csv import convert_csv

import pandas_datareader as web
import numpy as np
import pandas as pd

import math

def stock_data(ticker, start_date, end_date):
    
    stock_data = web.DataReader(ticker, data_source = 'yahoo', start = start_date, end = end_date )

    print(stock_data.head())
    
    return stock_data

def get_train_data(prediction_window, train_data, predict_col_num, num_of_data):
    
    train_len = len(train_data)
    train_vec_len = train_len - prediction_window
    #num_of_data = train_data.shape[1]
    
    X_train = np.zeros((train_vec_len, prediction_window, num_of_data))
    y_train = np.zeros((train_vec_len, 1))
    
    for i in range(0, train_vec_len):
        
        y_train[i] = train_data[i + prediction_window, predict_col_num]
        
        for j in range(0, prediction_window):
            
            for k in range(0, num_of_data):
                
                X_train[i, j, k] = train_data[i + j, k]
        
    return X_train, y_train 

def get_test_data(prediction_window, training_data_len, test_data, target, predict_col_num, num_of_data):
    
    test_len = len(test_data)
    test_vec_len = test_len - prediction_window
    #num_of_data = test_data.shape[1]
    
    X_test = np.zeros((test_vec_len, prediction_window, num_of_data))
    y_test =  np.zeros((test_vec_len, 1))
    
    for i in range(0, test_vec_len):
        
        y_test[i] = test_data[i + prediction_window, predict_col_num]
        
        for j in range(0, prediction_window):
            
            for k in range(0, num_of_data):
                
                X_test[i, j, k] = test_data[i + j, k]
        
    return X_test, y_test 

def data_preprocessing(training_set_len, prediction_window, ticker, start_date, end_date, with_temp, plot_the_data):
    
    # Weather Data Location
    weather_filename = '../data/boundary_county.csv'
    
    # Pull Stock Data
    if not with_temp:
        
        all_stock_data = stock_data(ticker, start_date, end_date)
        all_data = all_stock_data
        
    else:
        
        # Pull the Data in the Form of the Dataframe
        filename, dates, weathers, all_stock_data = convert_csv(weather_filename, ticker, start_date, end_date)

        # Temp Dataframes
        

        # Put Appropiate Dataframes Together
        all_data = pd.concat([dates, all_stock_data, weathers], axis=1)
        all_stock_data = pd.concat([dates, all_stock_data], axis=1) 
        weathers = pd.concat([dates, weathers], axis=1)         
        
        # Get Index to be Date        
        all_data.set_index('DATE', inplace=True, drop=True, append=False)
        all_stock_data.set_index('DATE', inplace=True, drop=True, append=False)
        weathers.set_index('DATE', inplace=True, drop=True, append=False)
        
        print(all_data.head())
        
        for col in all_data.columns:
            print(col)
    
    # Plot Data in all_data
    if plot_the_data == True:    
        plot_data(all_stock_data)
        
    if with_temp == True and plot_the_data == True:
        plot_data(weathers)
    
    # Create a Dataframe With Only the Close Stock Price Column
    close_values = all_data.filter(['Close'])
    data_target = all_data
 
    # Convert the dataframe to a numpy array to train the LSTM model
    target = data_target.values
    close_target = close_values.values
    training_data_len = math.ceil(len(target)* training_set_len)
    
    # Get Number of Data
    num_data = target.shape[1]
    
    # Scaling
    sc = MinMaxScaler(feature_range=(0,1))
    close_scale = MinMaxScaler(feature_range=(0,1))
    
    training_scaled_data = sc.fit_transform(target)
    close_training_scaled_data = close_scale.fit_transform(close_target)

    train_data = training_scaled_data[0:training_data_len, : ]

    # Getting the predicted stock price
    test_data = training_scaled_data[training_data_len - prediction_window: , : ]

    return data_target, target, train_data, test_data, training_data_len, sc, close_scale, close_values, num_data





