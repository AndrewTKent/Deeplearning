from preprocess import data_preprocessing, get_train_data, get_test_data
from rnn_model import LSTM_Model, CNN_LSTM_Model, CNN_SelfAtten_LSTM_Model, train_model, stock_forcast
from plotting import visualize_stock_fit, visualize_loss, visualize_val_loss
from multi_plotting import multi_visualize_stock_fit, mutli_visualize_loss, mutli_visualize_val_loss
from utilities import print_shape, calculate_L_scores

import numpy as np

import sys
    
def run_model(prediction_parameters, ticker, dates, specifications, model_type):

    selfpredict = specifications[0]
    with_temp = specifications[1]
    plot_the_data = specifications[2]
    plot_other_predictions = specifications[3]
    visualize_results = specifications[4]
    return_L2 = specifications[5]
    visualize_loss = specifications[6]
    visualize_val_loss = specifications[7]

    # Important Parameters 
    epoch_num = prediction_parameters[0]
    batch_num = prediction_parameters[1]
    training_set_len = prediction_parameters[2]
    prediction_window = prediction_parameters[3]
    predict_col_num = prediction_parameters[4]
    prediction_num = prediction_parameters[5]
    
    # Date Parameters
    start_date = dates[0] 
    end_date = dates[1]
    
    # Process Data
    data_target, target, train_data, test_data, training_data_len, sc, close_scale, close_values, num_data = data_preprocessing(
        training_set_len, prediction_window, ticker, start_date, end_date, with_temp, plot_the_data)

    # Get Test and Train Data
    X_train, y_train = get_train_data(prediction_window, train_data, predict_col_num, num_data)
    X_test, y_test = get_test_data(prediction_window, training_data_len, test_data, target, predict_col_num, num_data)
    
    # Print the Shape of the Arrays
    print_shape(data_target, train_data, test_data, X_train, y_train, X_test, y_test)

    # Train Model Parameters
    input_shape = (X_train.shape[1], num_data)
    validation = (X_test, y_test)
    
    # Vectors for Each Prediction
    prediction_len = X_test.shape[0]
    multiple_predictions = np.zeros((prediction_len, prediction_num))
    multiple_loss = np.zeros((epoch_num, prediction_num))
    multiple_val_loss = np.zeros((epoch_num, prediction_num))
    
    # Begin Training and Instantiating Model
    if model_type == 'lstm':
        model = LSTM_Model(input_shape)
        model_label = 'LSTM'
        
    elif model_type == 'cnn_lstm':
        model = CNN_LSTM_Model(input_shape)
        model_label = 'CNN-LSTM'
        
    elif model_type == 'cnn_lstm_attn':
        model = CNN_SelfAtten_LSTM_Model(input_shape)
        model_label = 'CNN-LSTM w/ Attn'
            
    else: 
        sys.exit('No Model Specified')
    
    # Run for the Number of Total Predictions
    for i in range(0, prediction_num): 
    
        history = train_model(model, input_shape, validation, X_train, y_train, epoch_num, batch_num, sc, num_data, i, prediction_num)

        #Making predictions using the test dataset
        predicted_stock_price = stock_forcast(model, X_test, close_scale, selfpredict, prediction_window)

        # Run the function to illustrate accuracy and loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        for j in range(0, prediction_len):
            
            multiple_predictions[j,i] = predicted_stock_price[j]
            
        for k in range(0,epoch_num):
            
            multiple_loss[k, i] = loss[k]
            multiple_val_loss[k, i] = val_loss[k]
            
            
    # Plotting Functions        
    if visualize_results == True:
        
        visualize_stock_fit(data_target, training_data_len, multiple_predictions, ticker, selfpredict, prediction_num, model_label, with_temp, plot_other_predictions, visualize_results, return_L2)
    
    if visualize_loss == True:
        
        visualize_loss(multiple_loss, prediction_num, model_label, with_temp, ticker)
    
    if visualize_val_loss == True:
        
        visualize_val_loss(multiple_val_loss, prediction_num, model_label, with_temp, ticker) 
    
    if return_L2 == True:
        
        L1_Distance, L2_Distance = calculate_L_scores(data_target, training_data_len, multiple_predictions, prediction_num)
        
        return L2_Distance
    
    
def run_multi_model(prediction_parameters, ticker, dates, specifications):   

    selfpredict = specifications[0]
    with_temp = specifications[1]
    plot_the_data = specifications[2]
    plot_other_predictions = specifications[3]
    visualize_results = specifications[4]
    return_L2 = specifications[5]

    # Important Parameters 
    epoch_num = prediction_parameters[0]
    batch_num = prediction_parameters[1]
    training_set_len = prediction_parameters[2]
    prediction_window = prediction_parameters[3]
    predict_col_num = prediction_parameters[4]
    prediction_num = prediction_parameters[5]
    
    # Date Parameters
    start_date = dates[0] 
    end_date = dates[1]
    
    # Process Data
    data_target, target, train_data, test_data, training_data_len, sc, close_scale, close_values, num_data = data_preprocessing(
        training_set_len, prediction_window, ticker, start_date, end_date, with_temp, plot_the_data)

    # Get Test and Train Data
    X_train, y_train = get_train_data(prediction_window, train_data, predict_col_num, num_data)
    X_test, y_test = get_test_data(prediction_window, training_data_len, test_data, target, predict_col_num, num_data)
    
    # Print the Shape of the Arrays
    print_shape(data_target, train_data, test_data, X_train, y_train, X_test, y_test)

    # Train Model Parameters
    input_shape = (X_train.shape[1], num_data)
    validation = (X_test, y_test)
    
    # Vectors for Each Prediction
    prediction_len = X_test.shape[0]
    
    # Different Model Arrays
    LSTM_multiple_predictions = np.zeros((prediction_len, prediction_num))
    LSTM_multiple_loss = np.zeros((epoch_num, prediction_num))
    LSTM_multiple_val_loss = np.zeros((epoch_num, prediction_num))
    
    CNN_multiple_predictions = np.zeros((prediction_len, prediction_num))
    CNN_multiple_loss = np.zeros((epoch_num, prediction_num))
    CNN_multiple_val_loss = np.zeros((epoch_num, prediction_num))
    
    ATTN_multiple_predictions = np.zeros((prediction_len, prediction_num))
    ATTN_multiple_loss = np.zeros((epoch_num, prediction_num))
    ATTN_multiple_val_loss = np.zeros((epoch_num, prediction_num))
    
    # Run for the Number of Total Predictions for LSTM
    model = LSTM_Model(input_shape)
    
    for i in range(0, prediction_num): 
        
        history = train_model(model, input_shape, validation, X_train, y_train, epoch_num, batch_num, sc, num_data, i, prediction_num)

        #Making predictions using the test dataset
        predicted_stock_price = stock_forcast(model, X_test, close_scale, selfpredict, prediction_window)

        # Run the function to illustrate accuracy and loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        for j in range(0, prediction_len):
            
            LSTM_multiple_predictions[j,i] = predicted_stock_price[j]
            
        for k in range(0,epoch_num):
            
            LSTM_multiple_loss[k, i] = loss[k]
            LSTM_multiple_val_loss[k, i] = val_loss[k]

    # Run for the Number of Total Predictions for CNN
    model = CNN_LSTM_Model(input_shape)
    
    for i in range(0, prediction_num): 
    
        history = train_model(model, input_shape, validation, X_train, y_train, epoch_num, batch_num, sc, num_data, i, prediction_num)

        #Making predictions using the test dataset
        predicted_stock_price = stock_forcast(model, X_test, close_scale, selfpredict, prediction_window)

        # Run the function to illustrate accuracy and loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        for j in range(0, prediction_len):
            
            CNN_multiple_predictions[j,i] = predicted_stock_price[j]
            
        for k in range(0,epoch_num):
            
            CNN_multiple_loss[k, i] = loss[k]
            CNN_multiple_val_loss[k, i] = val_loss[k]
        
    # Run for the Number of Total Predictions for ATTN
    model = CNN_SelfAtten_LSTM_Model(input_shape)
    
    for i in range(0, prediction_num): 
    
        history = train_model(model, input_shape, validation, X_train, y_train, epoch_num, batch_num, sc, num_data, i, prediction_num)

        #Making predictions using the test dataset
        predicted_stock_price = stock_forcast(model, X_test, close_scale, selfpredict, prediction_window)

        # Run the function to illustrate accuracy and loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        for j in range(0, prediction_len):
            
            ATTN_multiple_predictions[j,i] = predicted_stock_price[j]
            
        for k in range(0,epoch_num):
            
            ATTN_multiple_loss[k, i] = loss[k]
            ATTN_multiple_val_loss[k, i] = val_loss[k]
        
    L2_Score = multi_visualize_stock_fit(data_target, training_data_len, LSTM_multiple_predictions, CNN_multiple_predictions, ATTN_multiple_predictions, ticker, selfpredict, prediction_num, with_temp, plot_other_predictions, visualize_results, return_L2)
    mutli_visualize_loss(LSTM_multiple_loss, CNN_multiple_loss, ATTN_multiple_loss, prediction_num, with_temp, ticker)
    mutli_visualize_val_loss(LSTM_multiple_val_loss, CNN_multiple_val_loss, ATTN_multiple_val_loss, prediction_num, with_temp, ticker)
    
    if return_L2 == True:
        
        return L2_Score


 
