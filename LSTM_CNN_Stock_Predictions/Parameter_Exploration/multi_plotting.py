import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utilities import accuracy, round_sig, calculate_multi_L_scores

pd.options.mode.chained_assignment = None

def multi_visualize_stock_fit(data_target, training_data_len, LSTM_multiple_predictions, CNN_multiple_predictions, ATTN_multiple_predictions, ticker, selfpredict, prediction_num, with_temp, plot_other_predictions, visualize_results, return_L2):
    
    if selfpredict == True: 
        prediction_legend_label = ' and w/ Selfpredict'
        
    else:
        prediction_legend_label = ''
        
    if with_temp == True:
        weather_label = ' w/ Weather Data' 
        
    else:
        weather_label = ' w/o Weather Data' 
    
    # Visualising the results
    train = data_target[:training_data_len]
    valid = data_target[training_data_len:]
    
    LSTM_prediction_average = np.zeros(LSTM_multiple_predictions.shape[0])
    CNN_prediction_average = np.zeros(CNN_multiple_predictions.shape[0])
    ATTN_prediction_average = np.zeros(ATTN_multiple_predictions.shape[0])
    
    for i in range(0, prediction_num):
        
        LSTM_prediction_lable = 'LSTM_Prediction_' + str(i) 
        valid[LSTM_prediction_lable] = LSTM_multiple_predictions[:,i]
        LSTM_prediction_average += LSTM_multiple_predictions[:,i]
        
        CNN_prediction_lable = 'CNN_Prediction_' + str(i) 
        valid[CNN_prediction_lable] = CNN_multiple_predictions[:,i]
        CNN_prediction_average += CNN_multiple_predictions[:,i]
        
        ATTN_prediction_lable = 'ATTN_Prediction_' + str(i) 
        valid[ATTN_prediction_lable] = ATTN_multiple_predictions[:,i]
        ATTN_prediction_average += ATTN_multiple_predictions[:,i]
        
    valid['LSTM Prediction Average'] = LSTM_prediction_average/prediction_num
    valid['CNN Prediction Average'] = CNN_prediction_average/prediction_num
    valid['ATTN Prediction Average'] = ATTN_prediction_average/prediction_num
        
    
    # Accuracy Score
    for i in range(0, prediction_num):
        
        LSTM_prediction_lable = 'LSTM_Prediction_' + str(i)
        CNN_prediction_lable = 'CNN_Prediction_' + str(i)
        ATTN_prediction_lable = 'ATTN_Prediction_' + str(i)
        
        LSTM_L1_Distance, LSTM_L2_Distance = accuracy(valid[['Close']], valid[[LSTM_prediction_lable]])
        CNN_L1_Distance, CNN_L2_Distance = accuracy(valid[['Close']], valid[[CNN_prediction_lable]])
        ATTN_L1_Distance, ATTN_L2_Distance = accuracy(valid[['Close']], valid[[ATTN_prediction_lable]])
        
        LSTM_L1_Distance += LSTM_L1_Distance
        LSTM_L2_Distance += LSTM_L2_Distance
        
        CNN_L1_Distance += CNN_L1_Distance
        CNN_L2_Distance += CNN_L2_Distance
        
        ATTN_L1_Distance += ATTN_L1_Distance
        ATTN_L2_Distance += ATTN_L2_Distance
    
    LSTM_L1_Distance, LSTM_L2_Distance = LSTM_L1_Distance/prediction_num, LSTM_L2_Distance/prediction_num
    LSTM_L1_Distance, LSTM_L2_Distance = round_sig(LSTM_L1_Distance), round_sig(LSTM_L2_Distance)
    
    CNN_L1_Distance, CNN_L2_Distance = CNN_L1_Distance/prediction_num, CNN_L2_Distance/prediction_num
    CNN_L1_Distance, CNN_L2_Distance = round_sig(CNN_L1_Distance), round_sig(CNN_L2_Distance)
    
    ATTN_L1_Distance, ATTN_L2_Distance = ATTN_L1_Distance/prediction_num, ATTN_L2_Distance/prediction_num
    ATTN_L1_Distance, ATTN_L2_Distance = round_sig(ATTN_L1_Distance), round_sig(ATTN_L2_Distance)
    
    LSTM_L1_Distance = str(LSTM_L1_Distance)
    LSTM_L2_Distance = str(LSTM_L2_Distance)
    
    CNN_L1_Distance = str(CNN_L1_Distance)
    CNN_L2_Distance = str(CNN_L2_Distance)
    
    ATTN_L1_Distance = str(ATTN_L1_Distance)
    ATTN_L2_Distance = str(ATTN_L2_Distance)

    # Plotting
    if visualize_results == True:
        
        plt.figure(figsize=(10,5))
        plt.title(' Model Forcasting for ' + ticker + weather_label + prediction_legend_label)
        plt.xlabel('DATE', fontsize=12)
        plt.ylabel('Close Price USD ($)', fontsize=12)
        
        if plot_other_predictions == True:
        
            for i in range(0, prediction_num):
            
                LSTM_prediction_lable = 'LSTM_Prediction_' + str(i) 
                CNN_prediction_lable = 'CNN_Prediction_' + str(i)
                ATTN_prediction_lable = 'ATTN_Prediction_' + str(i)
                
                plt.plot(valid[[LSTM_prediction_lable]], alpha=0.1)
                plt.plot(valid[[CNN_prediction_lable]], alpha=0.1)
                plt.plot(valid[[ATTN_prediction_lable]], alpha=0.1)
            
        
        plt.plot(valid['LSTM Prediction Average'], label = 'LSTM Average Prediction: ' + r'$L_1$ = {}'.format(LSTM_L1_Distance) + r', $L_2$ = {}'.format(LSTM_L2_Distance))
        plt.plot(valid['CNN Prediction Average'], label = 'CNN Average Prediction: ' + r'$L_1$ = {}'.format(CNN_L1_Distance) + r', $L_2$ = {}'.format(CNN_L2_Distance))
        plt.plot(valid['ATTN Prediction Average'], label = 'ATTN Average Prediction: ' + r'$L_1$ = {}'.format(ATTN_L1_Distance) + r', $L_2$ = {}'.format(ATTN_L2_Distance))
        
        plt.plot(train['Close'], label = 'Train')
        plt.plot(valid['Close'], label = 'Test')
        
        plt.legend(loc='best')
            
        plt.show()
        
    return LSTM_L2_Distance, CNN_L2_Distance, ATTN_L2_Distance
    
    
# Utility function for plotting of the model results
def mutli_visualize_loss(LSTM_multiple_loss, CNN_multiple_loss, ATTN_multiple_loss, prediction_num, with_temp, ticker):
  
    if with_temp == True:
        weather_label = ' w/ Weather Data' 
        
    else:
        weather_label = ' w/o Weather Data'
  
    # Plot the accuracy and loss curves
    epochs = len(LSTM_multiple_loss)
    epochs_axis= range(epochs)
    
    LSTM_Loss_average = np.zeros(epochs)
    CNN_Loss_average = np.zeros(epochs)
    ATTN_Loss_average = np.zeros(epochs)
    
    print(LSTM_Loss_average.shape)
    print(LSTM_multiple_loss.shape)
    print(LSTM_multiple_loss[:,0].shape)
    
    plt.figure(figsize=(7,5))
    
    for i in range(0, prediction_num):
        
        LSTM_Loss_average += LSTM_multiple_loss[:,i]
        CNN_Loss_average += CNN_multiple_loss[:,i]
        ATTN_Loss_average += ATTN_multiple_loss[:,i]
        
    LSTM_Loss_average = LSTM_Loss_average/prediction_num
    CNN_Loss_average = CNN_Loss_average/prediction_num
    ATTN_Loss_average = ATTN_Loss_average/ATTN_Loss_average
            
    plt.plot(epochs_axis, LSTM_Loss_average, label='LSTM Loss Average')
    plt.plot(epochs_axis, CNN_Loss_average, label='CNN Loss Average')
    plt.plot(epochs_axis, ATTN_Loss_average, label='ATTN Loss Average')
    
    plt.title(' Training Loss for ' + ticker + weather_label)
    plt.legend()
    plt.show()
    
    
def mutli_visualize_val_loss(LSTM_multiple_val_loss, CNN_multiple_val_loss, ATTN_multiple_val_loss, prediction_num, with_temp, ticker):
  
    if with_temp == True:
        weather_label = ' w/ Weather Data' 
        
    else:
        weather_label = ' w/o Weather Data'   
  
    # Plot the accuracy and loss curves
    epochs = len(LSTM_multiple_val_loss)
    epochs_axis= range(epochs)
    
    LSTM_Val_Loss_average = np.zeros(epochs)
    CNN_Val_Loss_average = np.zeros(epochs)
    ATTN_Val_Loss_average = np.zeros(epochs)
    
    plt.figure(figsize=(7,5))
    
    for i in range(0, prediction_num):
        
        LSTM_Val_Loss_average += LSTM_multiple_val_loss[:,i]
        CNN_Val_Loss_average += CNN_multiple_val_loss[:,i]
        ATTN_Val_Loss_average += ATTN_multiple_val_loss[:,i]
        
    LSTM_Val_Loss_average = LSTM_Val_Loss_average/prediction_num
    CNN_Val_Loss_average = CNN_Val_Loss_average/prediction_num
    ATTN_Val_Loss_average = ATTN_Val_Loss_average/prediction_num
            
    plt.plot(epochs_axis, LSTM_Val_Loss_average, label='LSTM Val Loss Average')
    plt.plot(epochs_axis, CNN_Val_Loss_average, label='CNN Val Loss Average')
    plt.plot(epochs_axis, ATTN_Val_Loss_average, label='ATTN Val Loss Average')
    
    plt.title(' Validation Loss for ' + ticker + weather_label)
    plt.legend()
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


