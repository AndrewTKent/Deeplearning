import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utilities import accuracy, round_sig, calculate_L_scores

pd.options.mode.chained_assignment = None

def visualize_stock_fit(data_target, training_data_len, predicted_stock_price, ticker, selfpredict, prediction_num, model_label, with_temp, plot_other_predictions, visualize_results, return_L2):
    
    L1_Distance, L2_Distance = calculate_L_scores(data_target, training_data_len, predicted_stock_price, prediction_num, return_L2)
    
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
    prediction_average = np.zeros(predicted_stock_price.shape[0])
    
    for i in range(0, prediction_num):
        
        prediction_lable = 'Prediction_' + str(i) 
        valid[prediction_lable] = predicted_stock_price[:,i]
        prediction_average += predicted_stock_price[:,i]
        
    valid['Prediction Average'] = prediction_average/prediction_num
        
    
    # Accuracy Score
    for i in range(0, prediction_num):
        
        prediction_lable = 'Prediction_' + str(i)    
    
    L1_Distance_Label = 'L1 Average Distance: ' + str(L1_Distance)
    L2_Distance_Label = 'L2 Average Distance: ' + str(L2_Distance)
    
    print('\n' + L1_Distance_Label)
    print('\n' + L2_Distance_Label)

    # Plotting
    plt.title(model_label +' Model Forcasting for ' + ticker + weather_label + prediction_legend_label)
    plt.xlabel('DATE', fontsize=12)
    plt.ylabel('Close Price USD ($)', fontsize=12)
    plt.plot(train['Close'], label = 'Train')
    plt.plot(valid['Close'], label = 'Test')
    #plt.legend(['Train', 'Test'], loc='upper left')
    
    
    if plot_other_predictions == True:
    
        for i in range(0, prediction_num):
        
            prediction_lable = 'Prediction_' + str(i)
            plt.plot(valid[[prediction_lable]], alpha=0.25)
        
    
    plt.plot(valid['Prediction Average'], label = 'Average Prediction: ' + r'$L_1$ = {}'.format(L1_Distance) + r', $L_2$ = {}'.format(L2_Distance))
    #plt.legend(['Average Prediction' + r'$L_1$ = {}'.format(L1_Distance) + r', $L_2$ = {}'.format(L2_Distance)], loc='upper left')
    plt.legend(loc='upper left')
    
        
    plt.show()
        
    
    
# Utility function for plotting of the model results
def visualize_loss(loss, prediction_num, model_label, with_temp, ticker):
  
    if with_temp == True:
        weather_label = ' w/ Weather Data' 
        
    else:
        weather_label = ' w/o Weather Data'   
  
    # Plot the accuracy and loss curves
    epochs = range(len(loss))
    
    plt.figure(figsize=(7,5))
    
    for i in range(0, prediction_num):
    
        plt.plot(epochs, loss[:,i], label=str(i))
    
    plt.title(model_label + ' Training Loss for ' + ticker + weather_label)
    plt.legend()
    plt.show()
    
    
def visualize_val_loss(val_loss, prediction_num, model_label, with_temp, ticker):
  
    if with_temp == True:
        weather_label = ' w/ Weather Data' 
        
    else:
        weather_label = ' w/o Weather Data'     
  
    # Plot the accuracy and loss curves
    epochs = range(len(val_loss))
    
    plt.figure(figsize=(7,5))
    
    for i in range(0, prediction_num):
    
        plt.plot(epochs, val_loss[:,i], label=str(i))
    
    plt.title(model_label + ' Validation Loss for ' + ticker + weather_label)
    plt.legend()
    plt.show()
    
def plot_data(dataset):

    values = dataset.values
    
    # specify columns to plot
    num_columns = len(dataset.columns)
    
    # plot each column
    plt.figure()
    
    for i in range(0, num_columns):
        
    	plt.subplot(num_columns, 1, i+1)
    	plt.plot(values[:, i])
    	plt.title(dataset.columns[i], y=0.5, loc='right')

    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


