from main_support import run_model, run_multi_model

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np

import pickle
import os 

def run(prediction_parameters, ticker, dates, specifications, models, which_run):
    
    # Important Parameters 
    batch_num_vector = prediction_parameters[1]
    prediction_window_vector = prediction_parameters[3]
    save_plot = specifications[8]
    epoch_num = prediction_parameters[0]
    prediction_num = prediction_parameters[5]
    
    # Make Vectors to Eventually Make Plot With 
    batch_num_len = batch_num_vector[1] - batch_num_vector[0]
    prediction_window_len = prediction_window_vector[1] - prediction_window_vector[0]
    
    # Make Vectors for Plotting
    batch_num_axis = np.zeros(batch_num_len).reshape(batch_num_len)
    prediction_window_axis = np.zeros(prediction_window_len).reshape(prediction_window_len)
    parameter_exploration_array = np.zeros(batch_num_len*prediction_window_len).reshape(batch_num_len, prediction_window_len)
    
    # Additional Parameters for Running
    single_run = which_run[0]
    multi_model = which_run[1]
    category_run = which_run[2]
    
    num_runs = batch_num_len * prediction_window_len
    
    for i in range(batch_num_vector[0], batch_num_vector[1]):
        
        batch_num_axis[i - batch_num_vector[0]] = i
        prediction_parameters[1] = i
        
        for j in range(prediction_window_vector[0], prediction_window_vector[1]):
            
            remaining_runs = num_runs - batch_num_len * (i - batch_num_vector[0]) - (j - prediction_window_vector[0])
            
            print('\n######### ' + str(remaining_runs) + ' of ' + str(num_runs) + ' Runs Remaining #########\n')
            
            prediction_window_axis[j - prediction_window_vector[0]] = j
            prediction_parameters[3] = j
    
            # Single Model Run
            if single_run == True:
                
                L2_Score = run_model(prediction_parameters, ticker, dates, specifications, models[2])
                
            # Run All Six Variations 
            if multi_model == True:    
                
                L2_Score = run_multi_model(prediction_parameters, ticker, dates, specifications)
            
            # Run the Model
            if category_run == True:
            
                for j in range(0, 2):
                
                    single_ticker = ticker[j]
                    
                    print('\n\n###################### Starting ' + single_ticker + ' Analysis ######################\n\n')
                    
                    for i in range(0,3):
                    
                        print('\n#########  Starting ' + models[i] + ' Run #########\n')
                        L2_Score += run_model(prediction_parameters, ticker, dates, specifications, models[i])
                
                L2_Score = L2_Score/len(ticker)
                        
            print(L2_Score)
            parameter_exploration_array[i - prediction_window_vector[0], j - prediction_window_vector[0]] = L2_Score
    
    # Plot the Parameter Surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    batch_num_axis, prediction_window_axis = np.meshgrid(batch_num_axis, prediction_window_axis)
    surf = ax.plot_surface(batch_num_axis, prediction_window_axis, parameter_exploration_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    ax.set_xlabel('$Batch Size$')
    ax.set_ylabel('$Window Size$')
    ax.set_zlabel(r'$L_2$')
    ax.set_title('Parameter Exploration')
    
    fig.colorbar(surf)
    plt.show()
    
    if save_plot == True:
        
        cwd = os.getcwd()
        cwd_head, cwd_tail = os.path.split(cwd)
        path = cwd_head + '/plots/parameter_exploration/'
        plot_name = ticker + '_epoch_' + str(epoch_num) + '_predictions_' + str(prediction_num) + '.fig.pickle'

        with open(path + plot_name, 'wb') as f:
            pickle.dump(fig, f)
            pickle.dump(fig, open(plot_name + '_hello', 'wb'))
    
    
    
    
    
    
    
    
    
    
    
    
    