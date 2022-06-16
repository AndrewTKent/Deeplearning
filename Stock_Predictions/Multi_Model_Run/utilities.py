from math import log10, floor
from datetime import datetime

import numpy as np

def calculate_multi_L_scores():
    
    return 

def calculate_L_scores(data_target, training_data_len, predicted_stock_price, prediction_num):
    
    # Visualising the results
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
        L1_Distance, L2_Distance = accuracy(valid[['Close']], valid[[prediction_lable]])
        
        L1_Distance += L1_Distance
        L2_Distance += L2_Distance
    
    L1_Distance, L2_Distance = L1_Distance/prediction_num, L2_Distance/prediction_num
    L1_Distance, L2_Distance = round_sig(L1_Distance), round_sig(L2_Distance)
    
    return L1_Distance, L2_Distance

def time(which_time):
    
    if which_time == 'start':
        time = datetime.now()
        print('\nStart Time: {}'.format(time))
        print('\n')
        return time
        
    if which_time == 'end':
        time = datetime.now()
        print('\nEnd Time: {}\n'.format(time))
        return time
        
    else:
        print('Run Time: {}'.format(which_time))
        
        
def get_train_data_old(prediction_window, train_data):
    
    X_train = []
    y_train = []
    
    for i in range(prediction_window, len(train_data)):
        
        X_train.append(train_data[i-prediction_window:i, 0])
        y_train.append(train_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
    return X_train, y_train 
        
def get_test_data_old(prediction_window, training_data_len, test_data, target):
    
    X_test = []
    y_test =  target[training_data_len : , : ]
    for i in range(prediction_window,len(test_data)):
        X_test.append(test_data[i-prediction_window:i,0])

    # Convert x_test to a numpy array
    X_test = np.array(X_test)

    #Reshape the data into the shape accepted by the LSTM
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        
    return X_test, y_test 


def print_shape(data_target, train_data, test_data, X_train, y_train, X_test, y_test):
    print('\n\ndata_target.shape: ', data_target.shape)
        
    print('\ntrain_data.shape: ', train_data.shape)
    print('\ntest_data.shape: ', test_data.shape)
    
    print('\nX_train.shape: ', X_train.shape)
    print('\ny_train.shape: ', y_train.shape)
    
    print('\nX_test.shape: ', X_test.shape)
    print('\ny_test.shape: ', y_test.shape)
    print('\n')
    
def accuracy(vec_1, vec_2):
    
    # Make Sure Vectors are the Right Type
    if type(vec_1) or type(vec_2) != 'numpy.ndarray':
        
        vec_1 = vec_1.values
        vec_2 = vec_2.values
    
    # Parameters
    vec_1_len = len(vec_1)
    vec_2_len = len(vec_2)
    
    # L2 Distance Numbers
    l1_distance = 0
    l1_distance_avg = 0
    l2_distance = 0
    l2_distance_avg = 0

    if vec_1_len != vec_2_len:
        
        print('Unable to Compute Accuracy, Vectors are Different Lengths')
        
        pass
    
    else:
        
        for i in range(0, vec_1_len):
            
            vec_1_norm = 1
            vec_2_norm = vec_2[i]/vec_1[i]
            
            l1_distance += (vec_2_norm - vec_1_norm)
            l2_distance += (vec_1_norm - vec_2_norm)**2
                    
        l1_distance_avg = l1_distance/vec_1_len
        l2_distance_avg = l2_distance/vec_2_len
    
    return l1_distance_avg[0], l2_distance_avg[0]

# Is NaN Function
def isNaN(num):
    return num != num
    
# Significant Figures Function
def round_sig(x, sig=3, small_value=1.0e-9):
    if isNaN(x) == True:
        return x
    else:
        return round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)
    











































    
