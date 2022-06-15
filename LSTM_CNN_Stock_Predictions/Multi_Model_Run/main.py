from utilities import time
from run import run

import os

def main():	 
    
    # Print Start Time
    start_time = time('start')     

    # Important Parameters
    epoch_num = 30
    batch_num = 10   
    training_set_len = .6
    prediction_window = 15     
    predict_col_num = 3 # Open Price is 3
    num_of_predictions = 1
    prediction_parameters = [epoch_num, batch_num, training_set_len, prediction_window, predict_col_num, num_of_predictions]
    
    # Tickers for Comparisons
    beer_tickers = ['BUD', 'TAP']
    tech_tickers = ['AAPL', 'GOOGL']
    bank_tickers = ['JPM', 'MS']
    pharma_tickers = ['JNJ', 'PFE']
    ticker = 'BUD'
    
    # Specify the Models Being Considered
    models = ['lstm', 'cnn_lstm', 'cnn_lstm_attn']
    
    # Ticker Data for Preprocessing
    start_date = '2012-1-1' 
    end_date = '2021-12-30'  
    dates = [start_date, end_date]
    
    # Additional Function Specifications
    self_predict = False
    with_weather_data = False
    plot_the_data = True
    plot_other_predictions = False
    specifications = [self_predict, with_weather_data, plot_the_data, plot_other_predictions]
    
    # Which Model to Run
    single_run = True
    multi_model = False
    category_run = False
    which_run = [single_run, multi_model, category_run]
    
    # Run the Models Specified
    run(prediction_parameters, ticker, dates, specifications, models, which_run)
    
    # Announce When Run is Over
    os.system('say "Finished."')
    
    # Print End and Run Time
    end_time = time('end')
    time(end_time - start_time)
    
if __name__ == '__main__':
	main()
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
