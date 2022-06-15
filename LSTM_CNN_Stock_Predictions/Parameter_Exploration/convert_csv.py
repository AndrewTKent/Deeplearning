import os
import pandas_datareader as web
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# returns new csvs date_weather_stock_filename, stock_filename, 
# and 3 dataframes: dates, weathers, stocks
def convert_csv(filename, ticker, start_date, end_date):
	df = pd.read_csv(filename, low_memory=False)
	return get_date_and_weather_daily_average(df, ticker, start_date, end_date)

def get_date_and_weather_daily_average(df, ticker, start_date, end_date):
	dates = df["DATE"]
	# convert date to float values
	date_only = pd.to_datetime(dates).dt.date
	df['DATE_FLOAT'] = pd.to_datetime(date_only).apply(lambda x: x.value)

	# NOTE: feel free to change these to any hourly data
	weathers = df[['HourlyPrecipitation', 
					'HourlyRelativeHumidity',
					'HourlyDewPointTemperature', 
					'HourlyDryBulbTemperature', 
					'HourlyWetBulbTemperature']]

	weathers.dropna(how='all', inplace=True)
	weathers.fillna(0.0, inplace=True)

	weathers["DATE_FLOAT"] = df["DATE_FLOAT"]

	# drop invalid weathers
	dates = dates[weathers["DATE_FLOAT"].index]
	
	# convert datetime to date: 2018-02-02
	dates = pd.to_datetime(dates).dt.date
	dates = dates.drop_duplicates()

	dates = pd.to_datetime(dates)
	dates = dates.astype(str)
	dates = dates + " 00:00:00"
	
	# print("Dates shape: {}".format(dates.shape))

	# first convert weathers df to float
	keepOnlyDigits(weathers)

	# calculate real temperature by dew point temp and relative humidity
	# reference: https://iridl.ldeo.columbia.edu/dochelp/QA/Basic/dewpoint.html
	weathers['HourlyRealTemperature'] = weathers['HourlyDewPointTemperature'] + ((100 - weathers['HourlyRelativeHumidity']) / 5)

	# then group by date so that we have date average for all weathers
	dew_point_temps = weathers[['DATE_FLOAT', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 
						'HourlyWetBulbTemperature', 'HourlyRealTemperature']]
	temps_high = dew_point_temps.groupby(['DATE_FLOAT']).max()
	temps_low = dew_point_temps.groupby(['DATE_FLOAT']).min()

	weathers = weathers.groupby(['DATE_FLOAT']).mean()

	weathers.index = dates.index
	temps_high.index = dates.index
	temps_low.index = dates.index

	# rename "Hourly" to "Daily"
	renamed = weathers.rename(columns=lambda x: "DailyAverage" + x[6:])
	temps_high = temps_high.rename(columns=lambda x: "DailyMax" + x[6:])
	temps_low = temps_low.rename(columns=lambda x: "DailyMin" + x[6:])

	weathers = pd.concat([temps_high, temps_low, renamed], axis=1)
	
	if ticker != '':
		if start_date < dates.iloc[0]:
			print("\nThere is no Data Before {}".format(dates.iloc[0]))
		start_date = max(start_date, dates.iloc[0])
		end_date = min(end_date, dates.iloc[-1])

		print("\nReturned Dates From {} to {}\n".format(start_date, end_date))

		stocks = stock_data(ticker, start_date, end_date)

		dates_set = []
		for i in dates.index:
			dates_set.append(dates[i])
		
		for i in dates.index:
			if dates[i][:10] not in stocks.index:
				dates.drop(i, inplace=True)
				weathers.drop(i, inplace=True)

		for date in stocks.index:
			if str(date) not in dates_set:
				# print("in stock not in date: {}".format(date))
				stocks.drop(date, inplace=True)

		stocks.index = dates.index
		filename = write_to_csv(dates, weathers, stocks, ticker)

	else:
		filename = write_to_csv(dates, weathers)

	dates = pd.to_datetime(dates)

	return filename, dates, weathers, stocks

def keepOnlyDigits(df):
	"""MIGHT NOT NEED THIS!!"""
	#for now, set any string elements to zero
	pd.options.mode.chained_assignment = None
	for column in df.columns:
		df[column] = pd.to_numeric(df[column], errors='coerce', downcast='float')
	df.fillna(0.0, inplace=True)
	
	#make sure they are all float
	return df.values.astype('float32')

def write_to_csv(dates, weathers, stocks=None, ticker=None):
	if not os.path.exists('../data/stock_weather/'):
		os.makedirs('/..data/stock_weather/')

	date_weather_stock_filename = '../data/stock_weather/date_weather_stock_data.csv'

	if ticker is None:
		df = pd.concat([dates, weathers], axis=1)
	else:
		df = pd.concat([dates, weathers, stocks], axis=1)
		date_weather_stock_filename = '../data/stock_weather/{}_date_weather_stock_data.csv'.format(ticker)

	if os.path.exists(date_weather_stock_filename):
		os.remove(date_weather_stock_filename)

	df.to_csv(date_weather_stock_filename)

	return date_weather_stock_filename

def stock_data(ticker, start_date, end_date):
	
	stock_data = web.DataReader(ticker, data_source = 'yahoo', start = start_date, end = end_date)
	
	return stock_data


# filename = '../../data/boundary_county.csv'
# ticker = 'BUD'
# start_date = '2015-1-1'
# end_date = '2021-12-30'

# convert_csv(filename, ticker, start_date, end_date)