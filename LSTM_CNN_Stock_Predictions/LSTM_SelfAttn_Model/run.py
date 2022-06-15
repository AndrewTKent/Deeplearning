from main_support import run_model, run_multi_model

def run(prediction_parameters, ticker, dates, specifications, models, which_run):
    
    single_run = which_run[0]
    multi_model = which_run[1]
    category_run = which_run[2]
    
    # Single Model Run
    if single_run == True:
        
        run_model(prediction_parameters, 'TAP', dates, specifications, 'cnn_lstm_attn')
        
    # Run All Six Variations 
    if single_run == True:    
        
        run_multi_model(prediction_parameters, 'BUD', dates, specifications)
    
    # Run the Model
    if category_run == True:
    
        for j in range(0, 2):
        
            single_ticker = ticker[j]
            
            print('\n\n###################### Starting ' + ticker + ' Analysis ######################\n\n')
            
            for i in range(0,3):
            
                print('\n#########  Starting ' + models[i] + ' Run #########\n')
                run_model(prediction_parameters, ticker, dates, specifications, models[i])